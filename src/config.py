from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


REPOVIDEO_HOME = Path(os.environ.get("REPOVIDEO_HOME", Path.home() / ".repovideo"))
LORA_DIR = REPOVIDEO_HOME / "loras"
CACHE_DIR = REPOVIDEO_HOME / "cache"
MODELS_DIR = REPOVIDEO_HOME / "models"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DOCKERFILES_DIR = TEMPLATES_DIR / "dockerfiles"

DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_DEMO_DURATION = 60
DEFAULT_ANECDOTE_DURATION = 8


class PipelineConfig(BaseModel):
    repo_url: str
    output_path: Path = Field(default_factory=lambda: Path("output.mp4"))
    demo_duration: int = DEFAULT_DEMO_DURATION
    no_anecdote: bool = False
    train_lora: Path | None = None
    lora_name: str | None = None
    model_size: str = "14B"  # "14B" or "1.3B" fallback
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    fps: int = DEFAULT_FPS
    use_colab: bool = False
    gdrive_path: Path | None = None
    anecdote_file: Path | None = None
    dataset_name: str | None = None
    remote_url: str | None = None  # URL of hosted inference server (NodeOps, etc.)


def ensure_dirs() -> None:
    for d in (REPOVIDEO_HOME, LORA_DIR, CACHE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
