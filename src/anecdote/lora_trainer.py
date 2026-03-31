"""Stage 2c: LoRA fine-tuning for Wan2.1 image-to-video model.

Proper video diffusion training loop:
  1. Load video clips + extract first frames
  2. Encode videos through the Wan VAE into latent space
  3. Encode text captions through the T5 text encoder
  4. Encode first-frame images through the CLIP image encoder
  5. Add noise via FlowMatch scheduler
  6. Forward through transformer, compute denoising loss
  7. Backprop through LoRA adapters only

Supports two modes:
  - Video mode (Tier 1/2 datasets): full video diffusion training with temporal learning
  - Image mode (Tier 3 datasets): image reconstruction training (style only, no motion)
"""

from __future__ import annotations

import gc
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import LORA_DIR, MODELS_DIR

console = Console()

WAN_I2V_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
WAN_I2V_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


@dataclass
class LoRATrainingConfig:
    reference_dir: Path | None = None
    dataset_name: str | None = None
    lora_name: str = "custom_style"
    rank: int = 32
    alpha: int = 32
    learning_rate: float = 1e-4
    num_train_steps: int = 500
    batch_size: int = 1
    resolution_width: int = 720
    resolution_height: int = 480
    num_frames: int = 17
    gradient_accumulation_steps: int = 4
    save_every_n_steps: int = 100
    seed: int = 42
    model_size: str = "14B"
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class TrainingDataset:
    """Prepared training data: triples of (first_frame, video_latents, caption)."""
    frame_paths: list[Path] = field(default_factory=list)
    video_paths: list[Path] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    tier: str = "video"


def train_lora(config: LoRATrainingConfig) -> Path:
    """Full LoRA training pipeline: prepare data -> configure -> train -> save."""
    if config.dataset_name and not config.reference_dir:
        from src.anecdote.datasets import download_dataset, get_dataset_tier
        console.print(f"[bold blue]Downloading[/] built-in dataset: {config.dataset_name}")
        config.reference_dir = download_dataset(config.dataset_name)

    if config.reference_dir is None:
        raise ValueError("Provide --train-lora <dir> or --dataset <name>")

    # Auto-detect VRAM and warn/downgrade for 720P on small GPUs
    if config.model_size == "14B" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
        if vram_gb < 45:
            console.print(
                f"[yellow]Warning:[/] 720P mode needs ~40 GB VRAM for training "
                f"(you have {vram_gb:.0f} GB). Switching to 480P mode."
            )
            config.model_size = "480P"
        elif vram_gb < 80:
            console.print(
                f"[dim]VRAM: {vram_gb:.0f} GB — tight for 720P, using staged loading[/]"
            )

    console.print(f"[bold blue]Starting LoRA training[/] from {config.reference_dir}")

    dataset = _prepare_dataset(config)
    if len(dataset.frame_paths) == 0:
        raise ValueError(f"No valid training data found in {config.reference_dir}")

    console.print(
        f"  [dim]Prepared {len(dataset.frame_paths)} samples "
        f"({len(dataset.video_paths)} with video, tier={dataset.tier})[/]"
    )

    output_dir = LORA_DIR / config.lora_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset.tier == "i2v":
        _run_i2v_training(config, dataset, output_dir)
    elif dataset.tier == "video" and len(dataset.video_paths) > 0:
        _run_video_training(config, dataset, output_dir)
    else:
        _run_image_training(config, dataset, output_dir)

    _save_training_metadata(config, dataset, output_dir)
    console.print(f"[bold green]LoRA training complete![/] Weights saved to {output_dir}")
    return output_dir


def _prepare_dataset(config: LoRATrainingConfig) -> TrainingDataset:
    """Build training pairs from the reference directory."""
    ref_dir = config.reference_dir
    dataset = TrainingDataset()

    # Determine tier from dataset marker or from the datasets module
    tier = "image"
    marker_file = ref_dir / ".downloaded"
    if marker_file.exists():
        try:
            meta = json.loads(marker_file.read_text())
            tier = meta.get("tier", "image")
        except (json.JSONDecodeError, KeyError):
            pass
    elif config.dataset_name:
        from src.anecdote.datasets import get_dataset_tier
        tier = get_dataset_tier(config.dataset_name)

    dataset.tier = tier

    saved_captions: dict[str, str] = {}
    captions_file = ref_dir / "captions.json"
    if captions_file.exists():
        saved_captions = json.loads(captions_file.read_text())

    for video_file in sorted(ref_dir.glob("*.mp4")) + sorted(ref_dir.glob("*.webm")):
        frame_path = ref_dir / f"frame_{video_file.stem}.png"
        if not frame_path.exists():
            frame_path = _extract_first_frame(video_file, config)
        if frame_path and frame_path.exists():
            dataset.video_paths.append(video_file)
            dataset.frame_paths.append(frame_path)
            dataset.captions.append(
                saved_captions.get(video_file.name, _generate_caption(video_file))
            )

    for img_file in sorted(ref_dir.glob("*.png")) + sorted(ref_dir.glob("*.jpg")) + sorted(ref_dir.glob("*.jpeg")):
        if img_file.stem.startswith("frame_"):
            continue
        already_paired = img_file in dataset.frame_paths
        if not already_paired:
            dataset.frame_paths.append(img_file)
            dataset.captions.append(
                saved_captions.get(img_file.name, _generate_caption(img_file))
            )

    return dataset


def _extract_first_frame(video_path: Path, config: LoRATrainingConfig) -> Path | None:
    """Extract the first frame from a video."""
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
    name = file_path.stem.replace("_", " ").replace("-", " ")
    return f"A video clip showing {name}, smooth motion, high quality"


# ---------------------------------------------------------------------------
# Video training: real diffusion denoising loss on VAE-encoded video latents
# ---------------------------------------------------------------------------

def _run_video_training(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    """Train LoRA with actual video data through the full Wan2.1 pipeline.

    Loads each encoder one at a time to avoid OOM on 40GB GPUs with 14B model.
    """
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel
    from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
    from peft import LoraConfig, get_peft_model
    import torchvision.transforms as T

    model_id = WAN_I2V_720P if config.model_size == "14B" else WAN_I2V_480P
    device = _get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    console.print(f"  [dim]Loading model components from {model_id} (staged)[/]")

    video_latents_cache: list[torch.Tensor] = []
    text_embeds_cache: list[torch.Tensor] = []
    image_embeds_cache: list[torch.Tensor] = []

    # --- Stage A: VAE — encode all videos into latents, then free ---
    console.print("  [dim]Stage A: Encoding videos through VAE...[/]")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        cache_dir=str(MODELS_DIR),
    )
    vae.to(device).eval().requires_grad_(False)

    vae_scaling = vae.config.get("scaling_factor", 1.0) if isinstance(vae.config, dict) else getattr(vae.config, "scaling_factor", 1.0)
    for i, video_path in enumerate(dataset.video_paths):
        video_tensor = _load_video_as_tensor(
            video_path, config.num_frames,
            config.resolution_height, config.resolution_width,
        )
        if video_tensor is None:
            video_latents_cache.append(None)
            continue

        video_tensor = video_tensor.to(device, dtype=torch.float32)
        video_input = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            video_latent = vae.encode(video_input).latent_dist.sample() * vae_scaling
        video_latents_cache.append(video_latent.squeeze(0).cpu())

        if (i + 1) % 10 == 0:
            console.print(f"    [dim]VAE-encoded {i+1}/{len(dataset.video_paths)} videos[/]")

    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage B: Text encoder — encode captions, then free ---
    console.print("  [dim]Stage B: Encoding captions through UMT5...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", cache_dir=str(MODELS_DIR),
    )
    from transformers import UMT5EncoderModel
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    text_encoder.to(device).eval().requires_grad_(False)

    for caption in dataset.captions[:len(dataset.video_paths)]:
        text_inputs = tokenizer(
            caption, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embed = text_encoder(text_inputs.input_ids).last_hidden_state
        text_embeds_cache.append(text_embed.squeeze(0).cpu())

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage C: CLIP — encode first frames, then free ---
    console.print("  [dim]Stage C: Encoding first frames through CLIP...[/]")
    image_processor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor", cache_dir=str(MODELS_DIR),
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32,
        cache_dir=str(MODELS_DIR),
    )
    image_encoder.to(device).eval().requires_grad_(False)

    for frame_path in dataset.frame_paths[:len(dataset.video_paths)]:
        frame_img = Image.open(frame_path).convert("RGB")
        clip_inputs = image_processor(images=frame_img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embed = image_encoder(**clip_inputs).last_hidden_state
        image_embeds_cache.append(image_embed.squeeze(0).cpu())

    del image_encoder, image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Filter out failed encodes
    valid = [(vl, te, ie) for vl, te, ie in zip(video_latents_cache, text_embeds_cache, image_embeds_cache) if vl is not None]
    if not valid:
        raise ValueError("No videos could be encoded. Check your dataset has valid .mp4 files.")
    video_latents_cache, text_embeds_cache, image_embeds_cache = zip(*valid)
    video_latents_cache = list(video_latents_cache)
    text_embeds_cache = list(text_embeds_cache)
    image_embeds_cache = list(image_embeds_cache)

    console.print(f"  [dim]Cached {len(video_latents_cache)} video latents[/]")

    # --- Stage D: Load transformer + LoRA for training ---
    console.print("  [dim]Stage D: Loading transformer for training...[/]")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=str(MODELS_DIR),
    )

    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    transformer.to(device)

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    console.print(f"  [dim]LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)[/]")

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_train_steps, eta_min=config.learning_rate * 0.1,
    )

    num_train_timesteps = scheduler.config.get("num_train_timesteps", 1000)

    # Training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[loss]:.4f}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("Training LoRA (video)", total=config.num_train_steps, loss=0.0)
        accum_loss = 0.0

        for step in range(config.num_train_steps):
            idx = step % len(video_latents_cache)

            latents = video_latents_cache[idx].unsqueeze(0).to(device, dtype=weight_dtype)
            text_emb = text_embeds_cache[idx].unsqueeze(0).to(device, dtype=weight_dtype)
            image_emb = image_embeds_cache[idx].unsqueeze(0).to(device, dtype=weight_dtype)

            # latents shape: [B, C, F, H, W] from VAE; transformer expects [B, F, C, H, W]
            latents = latents.permute(0, 2, 1, 3, 4)
            batch_size, num_frames, num_channels, height, width = latents.shape

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0, num_train_timesteps, (batch_size,), device=device,
            ).long()

            # Flow matching: noisy = (1 - sigma) * x + sigma * noise
            # where sigma = timestep / num_train_timesteps
            sigmas = timesteps.float() / num_train_timesteps
            sigmas = sigmas.view(-1, 1, 1, 1, 1)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # I2V conditioning: pass first-frame CLIP embeddings so the model
            # learns to generate video conditioned on the source image
            model_output = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_emb,
                encoder_hidden_states_image=image_emb,
                timestep=timesteps,
                return_dict=False,
            )[0]

            target = latents - noise
            loss = torch.nn.functional.mse_loss(model_output, target)

            loss_val = loss.item()
            accum_loss += loss_val

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    transformer.parameters(), config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if (step + 1) % config.save_every_n_steps == 0:
                avg_loss = accum_loss / config.save_every_n_steps
                accum_loss = 0.0
                checkpoint_dir = output_dir / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(exist_ok=True)
                transformer.save_pretrained(checkpoint_dir)
                console.print(
                    f"  [dim]Checkpoint step={step+1}, avg_loss={avg_loss:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}[/]"
                )

            progress.update(task, advance=1, loss=loss_val)

    transformer.save_pretrained(output_dir)

    del transformer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# I2V training: proper image-to-video conditioning (TIP-I2V dataset)
# ---------------------------------------------------------------------------

def _run_i2v_training(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    """Train LoRA with image-to-video conditioning using image+text pairs.

    Uses image prompts as both the CLIP conditioning input and the VAE
    denoising target. The transformer learns to generate content conditioned
    on the source image via encoder_hidden_states_image.

    Loads each encoder one at a time to avoid OOM on 40GB GPUs with 14B model.
    """
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel
    from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
    from peft import LoraConfig, get_peft_model
    import torchvision.transforms as T

    model_id = WAN_I2V_720P if config.model_size == "14B" else WAN_I2V_480P
    device = _get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    console.print(f"  [dim]Loading I2V model components from {model_id} (staged)[/]")

    latents_cache: list[torch.Tensor] = []
    text_cache: list[torch.Tensor] = []
    image_embed_cache: list[torch.Tensor] = []

    image_transform = T.Compose([
        T.ToTensor(),
        T.Resize((config.resolution_height, config.resolution_width), antialias=True),
        T.Normalize([0.5], [0.5]),
    ])

    # --- Stage A: VAE encoding (load VAE, encode all images, free VAE) ---
    console.print("  [dim]Stage A: Encoding images through VAE...[/]")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        cache_dir=str(MODELS_DIR),
    )
    vae.to(device).eval().requires_grad_(False)

    vae_scaling = vae.config.get("scaling_factor", 1.0) if isinstance(vae.config, dict) else getattr(vae.config, "scaling_factor", 1.0)
    for frame_path in dataset.frame_paths:
        img = Image.open(frame_path).convert("RGB")
        img_tensor = image_transform(img).to(device)
        img_input = img_tensor.unsqueeze(0).unsqueeze(2)  # [B, C, 1, H, W]
        with torch.no_grad():
            latent = vae.encode(img_input).latent_dist.sample() * vae_scaling
        latents_cache.append(latent.squeeze(0).cpu())

    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print(f"    [dim]{len(latents_cache)} VAE latents cached[/]")

    # --- Stage B: Text encoding (load tokenizer + UMT5, encode captions, free) ---
    console.print("  [dim]Stage B: Encoding captions through UMT5...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", cache_dir=str(MODELS_DIR),
    )
    from transformers import UMT5EncoderModel
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    text_encoder.to(device).eval().requires_grad_(False)

    for caption in dataset.captions:
        text_inputs = tokenizer(
            caption, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embed = text_encoder(text_inputs.input_ids).last_hidden_state
        text_cache.append(text_embed.squeeze(0).cpu())

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print(f"    [dim]{len(text_cache)} text embeddings cached[/]")

    # --- Stage C: CLIP image encoding (load CLIP, encode images, free) ---
    console.print("  [dim]Stage C: Encoding images through CLIP...[/]")
    image_processor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor", cache_dir=str(MODELS_DIR),
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32,
        cache_dir=str(MODELS_DIR),
    )
    image_encoder.to(device).eval().requires_grad_(False)

    for frame_path in dataset.frame_paths:
        img = Image.open(frame_path).convert("RGB")
        clip_inputs = image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embed = image_encoder(**clip_inputs).last_hidden_state
        image_embed_cache.append(image_embed.squeeze(0).cpu())

    del image_encoder, image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print(f"    [dim]{len(image_embed_cache)} CLIP embeddings cached[/]")

    # --- Stage D: Load transformer + LoRA for training ---
    console.print("  [dim]Stage D: Loading transformer for training...[/]")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=str(MODELS_DIR),
    )

    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    transformer.to(device)

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    console.print(f"  [dim]LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)[/]")

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    console.print(f"  [dim]Cached {len(latents_cache)} I2V training pairs[/]")

    num_train_timesteps = scheduler.config.get("num_train_timesteps", 1000)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[loss]:.4f}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("Training LoRA (I2V)", total=config.num_train_steps, loss=0.0)

        for step in range(config.num_train_steps):
            idx = step % len(latents_cache)

            latents = latents_cache[idx].unsqueeze(0).to(device, dtype=dtype)
            text_emb = text_cache[idx].unsqueeze(0).to(device, dtype=dtype)
            image_emb = image_embed_cache[idx].unsqueeze(0).to(device, dtype=dtype)

            # [B, C, 1, H, W] -> [B, 1, C, H, W]
            latents = latents.permute(0, 2, 1, 3, 4)
            batch_size = latents.shape[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()

            sigmas = timesteps.float() / num_train_timesteps
            sigmas = sigmas.view(-1, 1, 1, 1, 1)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            model_output = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_emb,
                encoder_hidden_states_image=image_emb,
                timestep=timesteps,
                return_dict=False,
            )[0]

            target = latents - noise
            loss = torch.nn.functional.mse_loss(model_output, target)
            loss_val = loss.item()

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % config.save_every_n_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(exist_ok=True)
                transformer.save_pretrained(checkpoint_dir)

            progress.update(task, advance=1, loss=loss_val)

    transformer.save_pretrained(output_dir)

    del transformer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image-only training: fallback for Tier 3 datasets (style only, no motion)
# ---------------------------------------------------------------------------

def _run_image_training(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    """Train LoRA on image data only (no video temporal learning).

    Uses image reconstruction: encode image through VAE as a single-frame
    "video", then train the denoising objective. Teaches visual style but
    not motion. Prints a clear warning about the limitation.
    """
    console.print(
        "[yellow]Warning:[/] Image-only training mode. The model will learn visual "
        "style but NOT motion/temporal coherence. Use a video dataset (pusa, cinematic, "
        "pexels) for full quality."
    )

    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import torchvision.transforms as T

    model_id = WAN_I2V_720P if config.model_size == "14B" else WAN_I2V_480P
    device = _get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    latents_cache: list[torch.Tensor] = []
    text_cache: list[torch.Tensor] = []

    image_transform = T.Compose([
        T.ToTensor(),
        T.Resize((config.resolution_height, config.resolution_width), antialias=True),
        T.Normalize([0.5], [0.5]),
    ])

    # --- Stage A: VAE — encode images as single-frame "videos", then free ---
    console.print("  [dim]Stage A: Encoding images through VAE...[/]")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        cache_dir=str(MODELS_DIR),
    )
    vae.to(device).eval().requires_grad_(False)

    vae_scaling = vae.config.get("scaling_factor", 1.0) if isinstance(vae.config, dict) else getattr(vae.config, "scaling_factor", 1.0)
    for frame_path in dataset.frame_paths:
        img = Image.open(frame_path).convert("RGB")
        img_tensor = image_transform(img).to(device)
        img_input = img_tensor.unsqueeze(0).unsqueeze(2)  # [B, C, 1, H, W]
        with torch.no_grad():
            latent = vae.encode(img_input).latent_dist.sample() * vae_scaling
        latents_cache.append(latent.squeeze(0).cpu())

    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage B: Text encoder — encode captions, then free ---
    console.print("  [dim]Stage B: Encoding captions through UMT5...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", cache_dir=str(MODELS_DIR),
    )
    from transformers import UMT5EncoderModel
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    text_encoder.to(device).eval().requires_grad_(False)

    for caption in dataset.captions:
        text_inputs = tokenizer(
            caption, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embed = text_encoder(text_inputs.input_ids).last_hidden_state
        text_cache.append(text_embed.squeeze(0).cpu())

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage C: Load transformer + LoRA for training ---
    console.print("  [dim]Stage C: Loading transformer for training...[/]")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=str(MODELS_DIR),
    )

    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype,
        cache_dir=str(MODELS_DIR),
    )
    transformer.to(device)

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    num_train_timesteps = scheduler.config.get("num_train_timesteps", 1000)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[loss]:.4f}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("Training LoRA (image)", total=config.num_train_steps, loss=0.0)

        for step in range(config.num_train_steps):
            idx = step % len(latents_cache)

            latents = latents_cache[idx].unsqueeze(0).to(device, dtype=dtype)
            text_emb = text_cache[idx].unsqueeze(0).to(device, dtype=dtype)

            # [B, C, 1, H, W] -> [B, 1, C, H, W]
            latents = latents.permute(0, 2, 1, 3, 4)
            batch_size = latents.shape[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device).long()

            sigmas = timesteps.float() / num_train_timesteps
            sigmas = sigmas.view(-1, 1, 1, 1, 1)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            model_output = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_emb,
                timestep=timesteps,
                return_dict=False,
            )[0]

            target = latents - noise
            loss = torch.nn.functional.mse_loss(model_output, target)
            loss_val = loss.item()

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % config.save_every_n_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step + 1}"
                checkpoint_dir.mkdir(exist_ok=True)
                transformer.save_pretrained(checkpoint_dir)

            progress.update(task, advance=1, loss=loss_val)

    transformer.save_pretrained(output_dir)

    del transformer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Video loading utilities
# ---------------------------------------------------------------------------

def _load_video_as_tensor(
    video_path: Path,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor | None:
    """Load a video file as a tensor of shape [F, C, H, W] normalized to [-1, 1].

    Tries decord first (fast), falls back to torchvision, then to ffmpeg+PIL.
    """
    try:
        return _load_video_decord(video_path, num_frames, height, width)
    except Exception:
        pass

    try:
        return _load_video_torchvision(video_path, num_frames, height, width)
    except Exception:
        pass

    try:
        return _load_video_ffmpeg(video_path, num_frames, height, width)
    except Exception:
        return None


def _load_video_decord(
    video_path: Path, num_frames: int, height: int, width: int,
) -> torch.Tensor:
    """Load video with decord (fastest option)."""
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(str(video_path), width=width, height=height)
    total_frames = len(vr)
    indices = _sample_frame_indices(total_frames, num_frames)
    frames = vr.get_batch(indices)  # [F, H, W, C]
    frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # normalize to [-1, 1]
    return frames


def _load_video_torchvision(
    video_path: Path, num_frames: int, height: int, width: int,
) -> torch.Tensor:
    """Load video with torchvision.io."""
    import torchvision
    import torchvision.transforms as T

    reader = torchvision.io.VideoReader(str(video_path), "video")
    frames_list = []
    for frame in reader:
        frames_list.append(frame["data"])
        if len(frames_list) >= num_frames * 3:
            break

    if not frames_list:
        raise ValueError("No frames read")

    all_frames = torch.stack(frames_list)  # [N, C, H, W]
    indices = _sample_frame_indices(len(all_frames), num_frames)
    selected = all_frames[indices]

    resize = T.Resize((height, width), antialias=True)
    selected = resize(selected)
    selected = selected.float() / 127.5 - 1.0
    return selected


def _load_video_ffmpeg(
    video_path: Path, num_frames: int, height: int, width: int,
) -> torch.Tensor:
    """Load video by extracting frames with ffmpeg (universal fallback)."""
    import tempfile
    import torchvision.transforms as T

    tmpdir = Path(tempfile.mkdtemp(prefix="rv_frames_"))
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale={width}:{height},fps=8",
        str(tmpdir / "frame_%04d.png"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    frame_files = sorted(tmpdir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError("ffmpeg extracted no frames")

    indices = _sample_frame_indices(len(frame_files), num_frames)
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    frames = []
    for idx in indices:
        img = Image.open(frame_files[idx]).convert("RGB")
        frames.append(transform(img))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return torch.stack(frames)  # [F, C, H, W]


def _sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Uniformly sample num_frames indices from total_frames."""
    if total_frames <= num_frames:
        return list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    step = total_frames / num_frames
    return [int(i * step) for i in range(num_frames)]


# ---------------------------------------------------------------------------
# Metadata & utilities
# ---------------------------------------------------------------------------

def _save_training_metadata(
    config: LoRATrainingConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> None:
    metadata = {
        "lora_name": config.lora_name,
        "rank": config.rank,
        "alpha": config.alpha,
        "learning_rate": config.learning_rate,
        "num_train_steps": config.num_train_steps,
        "model_size": config.model_size,
        "num_frames": config.num_frames,
        "tier": dataset.tier,
        "num_video_samples": len(dataset.video_paths),
        "num_image_samples": len(dataset.frame_paths) - len(dataset.video_paths),
        "total_samples": len(dataset.frame_paths),
        "training_captions": dataset.captions[:20],
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))


def _find_latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        for name in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors"):
            weights = checkpoints[-1] / name
            if weights.exists():
                return weights
    return output_dir


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
