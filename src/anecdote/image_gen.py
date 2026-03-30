"""Stage 2a: SDXL keyframe image generation for anecdote intros."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console

from src.config import MODELS_DIR

console = Console()

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, disfigured, bad anatomy, "
    "watermark, text, logo, ugly, duplicate, morbid, mutilated, out of frame, "
    "extra limbs, bad proportions, grainy, cartoon, anime, illustration"
)


def generate_keyframes(
    prompts: list[str],
    output_dir: Path | None = None,
    width: int = 1280,
    height: int = 720,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> list[Path]:
    """Generate keyframe images from text prompts using SDXL."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="repovideo_keyframes_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Generating[/] {len(prompts)} keyframe images with SDXL")

    pipe = _load_sdxl_pipeline()
    image_paths: list[Path] = []

    for i, prompt in enumerate(prompts):
        console.print(f"  [dim]Keyframe {i+1}/{len(prompts)}:[/] {prompt[:80]}...")

        enhanced_prompt = _enhance_prompt(prompt)

        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        path = output_dir / f"keyframe_{i:03d}.png"
        image.save(path)
        image_paths.append(path)
        console.print(f"  [green]Saved[/] {path.name}")

    _unload_pipeline(pipe)
    return image_paths


def _load_sdxl_pipeline():
    """Load the SDXL pipeline with appropriate device and dtype settings."""
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    console.print(f"  [dim]Loading SDXL on {device} ({dtype})[/]")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=str(MODELS_DIR),
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def _enhance_prompt(prompt: str) -> str:
    """Add quality boosters to the prompt."""
    quality_suffix = ", masterpiece, best quality, highly detailed, sharp focus, professional photography"
    if not any(q in prompt.lower() for q in ["4k", "8k", "masterpiece", "best quality"]):
        return prompt + quality_suffix
    return prompt


def _unload_pipeline(pipe) -> None:
    """Free GPU memory."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
