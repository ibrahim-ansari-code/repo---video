"""Remote GPU dispatcher — offload GPU work to Google Colab via Google Drive sync.

Flow:
1. Local CLI writes a job (config.json + data) to Google Drive
2. Colab notebook picks it up, runs GPU inference, writes results
3. Local CLI polls for completion and downloads results

Requires: Google Drive desktop app syncing ~/Google Drive/
or manual gdrive path configuration.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

DEFAULT_GDRIVE_PATHS = [
    Path.home() / "Google Drive" / "My Drive",
    Path.home() / "Google Drive",
    Path.home() / "Library" / "CloudStorage" / "GoogleDrive-" ,  # macOS pattern
    Path("/Volumes/GoogleDrive/My Drive"),
]

JOBS_FOLDER_NAME = "repovideo_jobs"


def find_gdrive() -> Path | None:
    """Auto-detect the local Google Drive sync folder."""
    for base in DEFAULT_GDRIVE_PATHS:
        if base.exists():
            return base

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.exists():
        for entry in cloud_storage.iterdir():
            if entry.name.startswith("GoogleDrive"):
                candidate = entry / "My Drive"
                if candidate.exists():
                    return candidate
                if entry.exists():
                    return entry

    return None


def get_jobs_dir(gdrive_path: Path | None = None) -> Path:
    """Get or create the repovideo jobs directory on Google Drive."""
    if gdrive_path is None:
        gdrive_path = find_gdrive()
    if gdrive_path is None:
        raise RuntimeError(
            "Google Drive not found. Install Google Drive for Desktop, "
            "or set --gdrive-path to your Drive sync folder."
        )
    jobs_dir = gdrive_path / JOBS_FOLDER_NAME
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def submit_anecdote_job(
    keyframe_prompts: list[str],
    motion_prompts: list[str],
    model_size: str = "14B",
    lora_name: str | None = None,
    gdrive_path: Path | None = None,
) -> str:
    """Submit an anecdote generation job to Colab via Google Drive."""
    jobs_dir = get_jobs_dir(gdrive_path)
    job_id = f"anecdote_{uuid.uuid4().hex[:8]}"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "job_type": "generate_anecdote",
        "job_id": job_id,
        "keyframe_prompts": keyframe_prompts,
        "motion_prompts": motion_prompts,
        "model_size": model_size,
        "lora_path": str(jobs_dir / "loras" / lora_name) if lora_name else None,
    }

    (job_dir / "config.json").write_text(json.dumps(config, indent=2))
    (job_dir / "status.txt").write_text("pending")

    console.print(f"[bold blue]Submitted[/] job [bold]{job_id}[/] to Google Drive")
    console.print(f"  [dim]Location: {job_dir}[/]")
    return job_id


def submit_lora_training_job(
    reference_dir: Path,
    lora_name: str = "custom_style",
    model_size: str = "14B",
    num_train_steps: int = 500,
    gdrive_path: Path | None = None,
) -> str:
    """Submit a LoRA training job to Colab via Google Drive."""
    jobs_dir = get_jobs_dir(gdrive_path)
    job_id = f"lora_{uuid.uuid4().hex[:8]}"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_dest = job_dir / "reference_data"
    shutil.copytree(reference_dir, ref_dest, dirs_exist_ok=True)

    config = {
        "job_type": "train_lora",
        "job_id": job_id,
        "lora_name": lora_name,
        "model_size": model_size,
        "num_train_steps": num_train_steps,
    }

    (job_dir / "config.json").write_text(json.dumps(config, indent=2))
    (job_dir / "status.txt").write_text("pending")

    console.print(f"[bold blue]Submitted[/] LoRA training job [bold]{job_id}[/]")
    console.print(f"  [dim]Copied {len(list(reference_dir.iterdir()))} files to Drive[/]")
    return job_id


def wait_for_job(job_id: str, timeout: int = 3600, gdrive_path: Path | None = None) -> Path:
    """Poll Google Drive until the Colab worker marks the job as completed."""
    jobs_dir = get_jobs_dir(gdrive_path)
    job_dir = jobs_dir / job_id
    status_file = job_dir / "status.txt"
    results_dir = job_dir / "results"

    console.print(f"[bold]Waiting for Colab[/] to process job [cyan]{job_id}[/]...")
    console.print("  [dim]Make sure the Colab notebook is running (Option B: watch_for_jobs)[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for Colab GPU worker...", total=None)
        start = time.time()

        while time.time() - start < timeout:
            if status_file.exists():
                status = status_file.read_text().strip()
                if status == "completed":
                    progress.update(task, description="[green]Job completed!")
                    break
                if status.startswith("failed"):
                    raise RuntimeError(f"Colab job failed: {status}")

            elapsed = int(time.time() - start)
            progress.update(task, description=f"Waiting for Colab GPU worker... ({elapsed}s)")
            time.sleep(10)
        else:
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    console.print(f"[bold green]Job complete![/] Results at {results_dir}")
    return results_dir


def download_anecdote_result(job_id: str, output_path: Path, gdrive_path: Path | None = None) -> Path:
    """Download the generated anecdote video from a completed job."""
    results_dir = wait_for_job(job_id, gdrive_path=gdrive_path)

    anecdote_file = results_dir / "anecdote.mp4"
    if not anecdote_file.exists():
        clips = sorted((results_dir / "clips").glob("*.mp4")) if (results_dir / "clips").exists() else []
        if clips:
            anecdote_file = clips[0]
        else:
            raise FileNotFoundError(f"No anecdote video found in {results_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(anecdote_file, output_path)
    console.print(f"[green]Downloaded[/] anecdote → {output_path}")
    return output_path


def download_lora_result(job_id: str, lora_name: str, gdrive_path: Path | None = None) -> Path:
    """Download trained LoRA weights from a completed job."""
    from src.config import LORA_DIR

    results_dir = wait_for_job(job_id, gdrive_path=gdrive_path)
    lora_src = results_dir / "lora"

    if not lora_src.exists():
        raise FileNotFoundError(f"No LoRA weights found in {results_dir}")

    lora_dest = LORA_DIR / lora_name
    shutil.copytree(lora_src, lora_dest, dirs_exist_ok=True)
    console.print(f"[green]Downloaded[/] LoRA → {lora_dest}")
    return lora_dest
