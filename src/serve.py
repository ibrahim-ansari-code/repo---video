"""Inference server — host Wan2.1 I2V + LoRA as a FastAPI endpoint.

Deploy this on a GPU machine (NodeOps, RunPod, etc.) and point the
repovideo CLI at it with --remote <url>.

Endpoints:
  POST /generate   — image + prompt → video (returns mp4 bytes)
  POST /keyframes  — text prompts → keyframe images (returns zip of PNGs)
  GET  /health     — returns model status, VRAM usage, loaded LoRA
  GET  /            — docs redirect

Usage:
  # Local dev:
  uvicorn src.serve:app --host 0.0.0.0 --port 8000

  # Docker:
  docker build -f Dockerfile.serve -t repovideo-inference .
  docker run --gpus all -p 8000:8000 repovideo-inference
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

# FastAPI is imported lazily at startup but we want the module to be importable
# even without it installed (for type checking, tests, etc.)
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response, StreamingResponse
except ImportError:
    FastAPI = None  # type: ignore

MODEL_SIZE = os.environ.get("REPOVIDEO_MODEL_SIZE", "1.3B")
LORA_PATH = os.environ.get("REPOVIDEO_LORA_PATH", "")
CACHE_DIR = os.environ.get("REPOVIDEO_CACHE_DIR", "/models")

WAN_14B_I2V = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
WAN_1_3B_I2V = "Wan-AI/Wan2.1-I2V-1.3B-720P-Diffusers"

_pipeline = None
_sdxl_pipeline = None
_model_info: dict = {}


def _get_model_id() -> str:
    return WAN_14B_I2V if MODEL_SIZE == "14B" else WAN_1_3B_I2V


def _load_i2v_pipeline():
    """Load the Wan2.1 I2V pipeline once at startup."""
    global _pipeline, _model_info

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    model_id = _get_model_id()
    print(f"[serve] Loading {model_id}...")

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder",
        torch_dtype=torch.float32, cache_dir=CACHE_DIR,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32, cache_dir=CACHE_DIR,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR,
    )

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except AttributeError:
            pass
    else:
        pipe = pipe.to("cpu")

    lora_loaded = None
    if LORA_PATH and Path(LORA_PATH).exists():
        print(f"[serve] Loading LoRA from {LORA_PATH}")
        pipe.load_lora_weights(LORA_PATH)
        lora_loaded = LORA_PATH

    _pipeline = pipe
    _model_info = {
        "model_id": model_id,
        "model_size": MODEL_SIZE,
        "lora": lora_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(f"[serve] Ready. {_model_info}")


def _load_sdxl_pipeline():
    """Load SDXL for keyframe generation (optional, loaded on first request)."""
    global _sdxl_pipeline

    from diffusers import StableDiffusionXLPipeline

    print("[serve] Loading SDXL for keyframe generation...")
    _sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    if torch.cuda.is_available():
        _sdxl_pipeline.enable_model_cpu_offload()
    print("[serve] SDXL ready.")


def create_app() -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("FastAPI not installed. pip install fastapi uvicorn python-multipart")

    app = FastAPI(
        title="repovideo inference",
        description="Wan2.1 I2V + LoRA inference endpoint for repovideo",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup():
        _load_i2v_pipeline()

    @app.get("/health")
    async def health():
        vram_used = 0.0
        vram_total = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_mem / 1024**3

        return {
            "status": "ready" if _pipeline is not None else "loading",
            **_model_info,
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 2),
            "sdxl_loaded": _sdxl_pipeline is not None,
        }

    @app.post("/generate")
    async def generate_video(
        image: UploadFile = File(...),
        prompt: str = Form("smooth camera motion, cinematic, high quality"),
        num_frames: int = Form(25),
        width: int = Form(720),
        height: int = Form(480),
        num_inference_steps: int = Form(30),
        guidance_scale: float = Form(5.0),
    ):
        """Generate a video from an input image + motion prompt.

        Returns the video as raw mp4 bytes.
        """
        if _pipeline is None:
            raise HTTPException(503, "Model is still loading")

        from diffusers.utils import export_to_video

        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((width, height))

        start = time.time()
        output = _pipeline(
            image=img,
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        elapsed = time.time() - start

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(output.frames[0], tmp.name, fps=16)

        video_bytes = Path(tmp.name).read_bytes()
        Path(tmp.name).unlink(missing_ok=True)

        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "X-Inference-Time": f"{elapsed:.2f}s",
                "X-Model": _model_info.get("model_id", ""),
            },
        )

    @app.post("/keyframes")
    async def generate_keyframes(
        prompts: str = Form(...),
        width: int = Form(720),
        height: int = Form(480),
    ):
        """Generate keyframe images from text prompts (comma-separated JSON array).

        Returns a zip file containing PNG keyframes.
        """
        import json
        import zipfile

        if _sdxl_pipeline is None:
            _load_sdxl_pipeline()

        prompt_list = json.loads(prompts)
        if not isinstance(prompt_list, list):
            raise HTTPException(400, "prompts must be a JSON array of strings")

        images: list[bytes] = []
        for p in prompt_list:
            result = _sdxl_pipeline(
                prompt=p,
                width=width,
                height=height,
                num_inference_steps=30,
            )
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            images.append(buf.getvalue())

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i, img_bytes in enumerate(images):
                zf.writestr(f"keyframe_{i:03d}.png", img_bytes)
        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=keyframes.zip"},
        )

    @app.post("/lora/load")
    async def load_lora(path: str = Form(...)):
        """Hot-load a LoRA adapter from a path on the server filesystem."""
        if _pipeline is None:
            raise HTTPException(503, "Model is still loading")

        lora_dir = Path(path)
        if not lora_dir.exists():
            raise HTTPException(404, f"LoRA not found at {path}")

        _pipeline.load_lora_weights(str(lora_dir))
        _model_info["lora"] = str(lora_dir)
        return {"status": "loaded", "lora": str(lora_dir)}

    @app.post("/lora/upload")
    async def upload_lora(
        file: UploadFile = File(...),
        name: str = Form("uploaded_lora"),
    ):
        """Upload LoRA weights (.zip) and load them into the running model."""
        if _pipeline is None:
            raise HTTPException(503, "Model is still loading")

        import zipfile

        lora_dir = Path(CACHE_DIR) / "loras" / name
        lora_dir.mkdir(parents=True, exist_ok=True)

        zip_bytes = await file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(lora_dir)

        _pipeline.load_lora_weights(str(lora_dir))
        _model_info["lora"] = str(lora_dir)
        return {"status": "loaded", "lora": str(lora_dir), "name": name}

    return app


app = create_app() if FastAPI is not None else None
