"""Stage 2b: Wan2.1 image-to-video inference pipeline with LoRA support."""

from __future__ import annotations

import gc
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console

from src.config import MODELS_DIR, LORA_DIR, DEFAULT_FPS

console = Console()

WAN_14B_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
WAN_1_3B_MODEL_ID = "Wan-AI/Wan2.1-I2V-1.3B-720P-Diffusers"


def generate_video_from_images(
    image_paths: list[Path],
    motion_prompts: list[str],
    output_path: Path,
    lora_path: Path | None = None,
    model_size: str = "14B",
    num_frames: int = 25,
    fps: int = DEFAULT_FPS,
    width: int = 720,
    height: int = 480,
) -> Path:
    """Generate short video clips from keyframe images, then concatenate them."""
    console.print(f"[bold blue]Generating[/] {len(image_paths)} video clips with Wan2.1 I2V")

    clip_dir = Path(tempfile.mkdtemp(prefix="repovideo_clips_"))
    clip_paths: list[Path] = []

    pipe = _load_wan_pipeline(model_size, lora_path)

    for i, (img_path, motion) in enumerate(zip(image_paths, motion_prompts)):
        console.print(f"  [dim]Clip {i+1}/{len(image_paths)}:[/] {motion[:60]}...")

        clip_path = clip_dir / f"clip_{i:03d}.mp4"
        _generate_single_clip(pipe, img_path, motion, clip_path, num_frames, fps, width, height)
        clip_paths.append(clip_path)
        console.print(f"  [green]Saved[/] {clip_path.name}")

    _unload_pipeline(pipe)

    _concatenate_clips(clip_paths, output_path)
    console.print(f"[bold green]Generated[/] anecdote video: {output_path}")
    return output_path


def _load_wan_pipeline(model_size: str, lora_path: Path | None):
    """Load the Wan2.1 I2V pipeline with optional LoRA weights."""
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    model_id = WAN_14B_MODEL_ID if model_size == "14B" else WAN_1_3B_MODEL_ID
    device = _get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    console.print(f"  [dim]Loading Wan2.1 {model_size} on {device}[/]")

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except AttributeError:
            pass
    else:
        pipe = pipe.to(device)

    if lora_path and lora_path.exists():
        console.print(f"  [dim]Loading LoRA weights from {lora_path}[/]")
        pipe.load_lora_weights(str(lora_path))

    return pipe


def _generate_single_clip(
    pipe,
    image_path: Path,
    prompt: str,
    output_path: Path,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Generate a single video clip from an image + motion prompt."""
    from diffusers.utils import export_to_video

    image = Image.open(image_path).convert("RGB").resize((width, height))

    output = pipe(
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=5.0,
    )

    export_to_video(output.frames[0], str(output_path), fps=fps)


def _concatenate_clips(clip_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple short clips into one video using ffmpeg."""
    if len(clip_paths) == 1:
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return

    concat_file = Path(tempfile.mktemp(suffix=".txt", prefix="repovideo_concat_"))
    with open(concat_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unload_pipeline(pipe) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
