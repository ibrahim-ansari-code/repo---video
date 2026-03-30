"""Built-in dataset support for LoRA training.

Downloads curated video/image datasets from Hugging Face so users don't need
to provide their own reference material. Three style presets:

  cinematic  — Pexels stock footage: professional camera work, smooth motion
  developer  — Developer/tech-themed clips: screens, typing, office settings
  dramatic   — High-contrast dramatic scenes: good for anecdote intros

Each downloads a small subset (50-200 samples) suitable for LoRA fine-tuning.
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import CACHE_DIR

console = Console()

BUILTIN_DATASETS = {
    "cinematic": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "description": "Professional stock footage — smooth camera motion, cinematic lighting",
        "filter_keywords": [
            "cinematic", "aerial", "slow motion", "timelapse", "landscape",
            "sunset", "city", "nature", "ocean", "clouds",
        ],
        "num_samples": 100,
    },
    "developer": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "description": "Developer/tech scenes — screens, typing, offices, code",
        "filter_keywords": [
            "computer", "laptop", "coding", "typing", "office", "desk",
            "programmer", "developer", "screen", "keyboard", "technology",
            "working", "startup", "meeting",
        ],
        "num_samples": 100,
    },
    "dramatic": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "description": "High-contrast dramatic scenes — great for problem-setup anecdotes",
        "filter_keywords": [
            "dramatic", "dark", "moody", "storm", "rain", "night",
            "silhouette", "shadow", "intense", "frustrated", "stressed",
            "worried", "alone", "struggle",
        ],
        "num_samples": 100,
    },
}


def list_builtin_datasets() -> list[dict]:
    """Return info about available built-in datasets."""
    return [
        {"name": name, "description": cfg["description"], "samples": cfg["num_samples"]}
        for name, cfg in BUILTIN_DATASETS.items()
    ]


def download_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> Path:
    """Download a built-in dataset and prepare it for LoRA training.

    Returns the path to a directory containing images and a captions.json file.
    """
    if dataset_name not in BUILTIN_DATASETS:
        available = ", ".join(BUILTIN_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    cfg = BUILTIN_DATASETS[dataset_name]
    num_samples = max_samples or cfg["num_samples"]

    if output_dir is None:
        output_dir = CACHE_DIR / "datasets" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".downloaded"
    if marker.exists():
        existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        if len(existing) >= num_samples // 2:
            console.print(f"  [dim]Using cached dataset: {output_dir} ({len(existing)} images)[/]")
            return output_dir

    console.print(f"[bold blue]Downloading[/] '{dataset_name}' dataset ({num_samples} samples)")

    _download_pexels_thumbnails(cfg, output_dir, num_samples)

    marker.write_text(json.dumps({"dataset": dataset_name, "samples": num_samples}))
    final_count = len(list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png")))
    console.print(f"[bold green]Downloaded[/] {final_count} training images to {output_dir}")
    return output_dir


def _download_pexels_thumbnails(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download thumbnail images from the Pexels-400k dataset, filtered by keywords."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    hf_dataset = cfg["hf_dataset"]
    keywords = cfg["filter_keywords"]

    console.print(f"  [dim]Loading {hf_dataset} metadata...[/]")
    ds = load_dataset(hf_dataset, split="train", streaming=True)

    collected = 0
    captions = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Filtering '{cfg['filter_keywords'][0]}' videos", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            title = (row.get("title") or row.get("Title") or "").lower()
            tags = (row.get("tags") or row.get("Tags") or "").lower()
            searchable = f"{title} {tags}"

            if not any(kw in searchable for kw in keywords):
                continue

            thumbnail = row.get("thumbnail") or row.get("Thumbnail")
            if thumbnail is None:
                continue

            try:
                if isinstance(thumbnail, str):
                    img = _download_image(thumbnail)
                elif isinstance(thumbnail, Image.Image):
                    img = thumbnail
                elif isinstance(thumbnail, dict) and "bytes" in thumbnail:
                    img = Image.open(io.BytesIO(thumbnail["bytes"]))
                else:
                    continue

                img = img.convert("RGB").resize((720, 480))
                filename = f"sample_{collected:04d}.jpg"
                img.save(output_dir / filename, "JPEG", quality=90)

                caption = title.strip() or f"a video clip, smooth motion, cinematic"
                captions[filename] = f"{caption}, smooth motion, high quality, cinematic"
                collected += 1
                progress.update(task, completed=collected)

            except Exception:
                continue

    captions_path = output_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))


def _download_image(url: str) -> Image.Image:
    """Download a single image from URL."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read()))


def prepare_builtin_dataset_for_training(dataset_name: str) -> tuple[Path, list[str]]:
    """Download dataset and return (directory, captions) ready for LoRA training."""
    output_dir = download_dataset(dataset_name)

    captions_file = output_dir / "captions.json"
    if captions_file.exists():
        captions = json.loads(captions_file.read_text())
    else:
        captions = {}
        for img in sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png")):
            captions[img.name] = f"A video clip showing {img.stem.replace('_', ' ')}, smooth motion, high quality"

    return output_dir, list(captions.values())
