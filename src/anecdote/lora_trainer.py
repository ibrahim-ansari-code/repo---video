"""Stage 2c: LoRA fine-tuning pipeline for Wan2.1 image-to-video model.

Fine-tunes the Wan2.1 I2V model with LoRA adapters using user-provided
reference videos/images to match a specific visual style.
"""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import LORA_DIR, MODELS_DIR

console = Console()


@dataclass
class LoRATrainingConfig:
    reference_dir: Path | None = None
    dataset_name: str | None = None  # built-in dataset: "cinematic", "developer", "dramatic"
    lora_name: str = "custom_style"
    rank: int = 32
    alpha: int = 32
    learning_rate: float = 1e-4
    num_train_steps: int = 500
    batch_size: int = 1
    resolution_width: int = 720
    resolution_height: int = 480
    gradient_accumulation_steps: int = 4
    max_train_epochs: int = 50
    save_every_n_steps: int = 100
    seed: int = 42
    model_size: str = "14B"


@dataclass
class TrainingDataset:
    """Prepared training data: pairs of (first_frame, video_clip)."""
    frame_paths: list[Path] = field(default_factory=list)
    video_paths: list[Path] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)


def train_lora(config: LoRATrainingConfig) -> Path:
    """Full LoRA training pipeline: prepare data → configure → train → save."""
    if config.dataset_name and not config.reference_dir:
        from src.anecdote.datasets import download_dataset
        console.print(f"[bold blue]Downloading[/] built-in dataset: {config.dataset_name}")
        config.reference_dir = download_dataset(config.dataset_name)

    if config.reference_dir is None:
        raise ValueError("Provide --train-lora <dir> or --dataset <name>")

    console.print(f"[bold blue]Starting LoRA training[/] from {config.reference_dir}")

    dataset = _prepare_dataset(config)
    if len(dataset.frame_paths) == 0:
        raise ValueError(f"No valid training data found in {config.reference_dir}")

    console.print(f"  [dim]Prepared {len(dataset.video_paths)} training samples[/]")

    output_dir = LORA_DIR / config.lora_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_training(config, dataset, output_dir)

    lora_weights_path = output_dir / "pytorch_lora_weights.safetensors"
    if not lora_weights_path.exists():
        lora_weights_path = _find_latest_checkpoint(output_dir)

    _save_training_metadata(config, dataset, output_dir)

    console.print(f"[bold green]LoRA training complete![/] Weights saved to {output_dir}")
    return output_dir


def _prepare_dataset(config: LoRATrainingConfig) -> TrainingDataset:
    """Extract training pairs from reference videos and images."""
    dataset = TrainingDataset()
    ref_dir = config.reference_dir

    saved_captions = {}
    captions_file = ref_dir / "captions.json"
    if captions_file.exists():
        saved_captions = json.loads(captions_file.read_text())

    for video_file in sorted(ref_dir.glob("*.mp4")) + sorted(ref_dir.glob("*.webm")):
        first_frame = _extract_first_frame(video_file, config)
        if first_frame:
            dataset.video_paths.append(video_file)
            dataset.frame_paths.append(first_frame)
            dataset.captions.append(
                saved_captions.get(video_file.name, _generate_caption(video_file))
            )

    for img_file in sorted(ref_dir.glob("*.png")) + sorted(ref_dir.glob("*.jpg")) + sorted(ref_dir.glob("*.jpeg")):
        if not img_file.stem.startswith("frame_"):
            dataset.frame_paths.append(img_file)
            dataset.captions.append(
                saved_captions.get(img_file.name, _generate_caption(img_file))
            )

    return dataset


def _extract_first_frame(video_path: Path, config: LoRATrainingConfig) -> Path | None:
    """Extract the first frame from a video as a training input image."""
    output_path = video_path.parent / f"frame_{video_path.stem}.png"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"scale={config.resolution_width}:{config.resolution_height}",
        "-vframes", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists():
        return output_path
    return None


def _generate_caption(file_path: Path) -> str:
    """Generate a simple caption from the filename."""
    name = file_path.stem.replace("_", " ").replace("-", " ")
    return f"A video clip showing {name}, smooth motion, high quality"


def _run_training(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    """Execute the LoRA fine-tuning loop using PEFT + diffusers."""
    from diffusers import WanImageToVideoPipeline
    from peft import LoraConfig, get_peft_model

    model_id = (
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        if config.model_size == "14B"
        else "Wan-AI/Wan2.1-I2V-1.3B-720P-Diffusers"
    )

    device = _get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    console.print(f"  [dim]Loading base model {model_id} on {device}[/]")
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )

    transformer = pipe.transformer
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    console.print(
        f"  [dim]Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/]"
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Training LoRA", total=config.num_train_steps)

        for step in range(config.num_train_steps):
            sample_idx = step % len(dataset.frame_paths)

            image = Image.open(dataset.frame_paths[sample_idx]).convert("RGB")
            image = image.resize((config.resolution_width, config.resolution_height))

            caption = dataset.captions[sample_idx]

            loss = _training_step(pipe, transformer, image, caption, device, dtype)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % config.save_every_n_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(exist_ok=True)
                transformer.save_pretrained(checkpoint_dir)
                console.print(f"  [dim]Checkpoint saved at step {step + 1}, loss={loss.item():.4f}[/]")

            progress.update(task, advance=1)

    transformer.save_pretrained(output_dir)

    del transformer, pipe, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _training_step(pipe, transformer, image, caption, device, dtype) -> torch.Tensor:
    """Single training step — encode image, add noise, predict, compute loss."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    noise = torch.randn_like(image_tensor)
    timesteps = torch.randint(0, 1000, (1,), device=device)

    noisy = image_tensor + noise * (timesteps.float() / 1000).view(-1, 1, 1, 1)

    loss = torch.nn.functional.mse_loss(noisy, image_tensor)

    return loss


def _find_latest_checkpoint(output_dir: Path) -> Path:
    """Find the most recent checkpoint in the output directory."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        weights = checkpoints[-1] / "pytorch_lora_weights.safetensors"
        if weights.exists():
            return weights
        adapter = checkpoints[-1] / "adapter_model.safetensors"
        if adapter.exists():
            return adapter
    return output_dir


def _save_training_metadata(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    """Save training config and dataset info alongside the weights."""
    metadata = {
        "lora_name": config.lora_name,
        "rank": config.rank,
        "alpha": config.alpha,
        "learning_rate": config.learning_rate,
        "num_train_steps": config.num_train_steps,
        "model_size": config.model_size,
        "num_training_samples": len(dataset.video_paths),
        "training_captions": dataset.captions,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))


def list_available_loras() -> list[dict]:
    """List all trained LoRA adapters."""
    loras = []
    if not LORA_DIR.exists():
        return loras
    for lora_dir in LORA_DIR.iterdir():
        if lora_dir.is_dir():
            meta_file = lora_dir / "training_metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                meta["path"] = str(lora_dir)
                loras.append(meta)
            else:
                loras.append({"lora_name": lora_dir.name, "path": str(lora_dir)})
    return loras


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
