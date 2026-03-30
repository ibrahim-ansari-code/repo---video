"""Remote GPU dispatcher — two modes:

1. **Google Drive/Colab** (--colab): sync jobs via Google Drive
2. **HTTP inference** (--remote <url>): call a hosted Wan2.1 endpoint
   (NodeOps CreateOS, RunPod, or any server running `src/serve.py`)
"""

from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


# ============================================================================
# HTTP remote inference (NodeOps / any hosted endpoint)
# ============================================================================

class RemoteInferenceClient:
    """Client for the repovideo FastAPI inference server (src/serve.py)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self) -> None:
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"{self.base_url}/health", timeout=10)
            data = json.loads(resp.read())
            console.print(f"  [green]Connected[/] to remote GPU: {data.get('model_id', '?')}")
            console.print(f"  [dim]Device: {data.get('device')}, VRAM: {data.get('vram_used_gb', '?')}/{data.get('vram_total_gb', '?')} GB[/]")
            if data.get("lora"):
                console.print(f"  [dim]LoRA loaded: {data['lora']}[/]")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach inference server at {self.base_url}/health — {e}\n"
                "Make sure the server is running (see Dockerfile.serve or src/serve.py)"
            )

    def generate_video(
        self,
        image_path: Path,
        prompt: str,
        output_path: Path,
        num_frames: int = 25,
        width: int = 720,
        height: int = 480,
    ) -> Path:
        """Send an image + prompt to the remote server, receive mp4 bytes back."""
        import urllib.request
        from urllib.parse import urlencode

        console.print(f"  [dim]Sending to remote GPU: {prompt[:60]}...[/]")

        img_bytes = image_path.read_bytes()

        boundary = uuid.uuid4().hex
        body = _build_multipart_body(boundary, {
            "prompt": prompt,
            "num_frames": str(num_frames),
            "width": str(width),
            "height": str(height),
        }, {"image": (image_path.name, img_bytes, "image/png")})

        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            video_bytes = resp.read()
            elapsed = time.time() - start

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)
        console.print(f"  [green]Received[/] {len(video_bytes)/1024:.0f} KB in {elapsed:.1f}s")
        return output_path

    def generate_keyframes(
        self,
        prompts: list[str],
        output_dir: Path,
        width: int = 720,
        height: int = 480,
    ) -> list[Path]:
        """Generate keyframe images on the remote server."""
        import urllib.request
        import zipfile

        boundary = uuid.uuid4().hex
        body = _build_multipart_body(boundary, {
            "prompts": json.dumps(prompts),
            "width": str(width),
            "height": str(height),
        }, {})

        req = urllib.request.Request(
            f"{self.base_url}/keyframes",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            zip_bytes = resp.read()

        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".png"):
                    dest = output_dir / name
                    dest.write_bytes(zf.read(name))
                    paths.append(dest)

        return paths

    def upload_lora(self, lora_dir: Path, name: str = "uploaded_lora") -> None:
        """Zip and upload LoRA weights to the remote server."""
        import urllib.request

        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for f in lora_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(lora_dir))
        zip_bytes = zip_buf.getvalue()

        boundary = uuid.uuid4().hex
        body = _build_multipart_body(
            boundary,
            {"name": name},
            {"file": ("lora.zip", zip_bytes, "application/zip")},
        )

        req = urllib.request.Request(
            f"{self.base_url}/lora/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            console.print(f"  [green]LoRA uploaded[/] → {data.get('lora', '?')}")


def _build_multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> bytes:
    """Build a multipart/form-data body without external deps."""
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    for key, (filename, data, content_type) in files.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


# ============================================================================
# Google Drive / Colab sync (existing functionality)
# ============================================================================

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
