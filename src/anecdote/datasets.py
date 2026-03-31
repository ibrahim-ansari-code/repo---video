"""Built-in dataset support for LoRA training.

Three tiers of training data, from best (video) to easiest (image-only):

  Tier 1 — Video datasets (actual mp4 clips + captions):
    "pusa"        — PusaV1 training set: 3,860 Wan-generated video-caption pairs
                    (gold standard for I2V LoRA fine-tuning on Wan2.1)
    "cinematic"   — CinematicT2vData/cinepile_captions: ~3.5k cinematic clips
                    from films with detailed scene descriptions

  Tier 2 — Video-from-URL datasets (downloads video URLs from metadata):
    "pexels"      — jovianzm/Pexels-400k: stock video thumbnails/URLs filtered
                    by visual style keywords (cinematic, dramatic, etc.)
    "youtube-commons" — PleIAs/YouTube-Commons: CC-BY YouTube transcripts + links;
                    streams parquet shards and uses yt-dlp to grab short clips (slow; small N recommended).

  Tier 3 — Image-only datasets (still images + synthetic captions):
    "developer"   — Pexels filtered for tech/dev scenes (keyboard, screen, code)
    "dramatic"    — Pexels filtered for moody/dramatic scenes

Video datasets produce proper (first_frame, video_clip, caption) triples for
real video-diffusion training. Image-only datasets degrade gracefully into
image-reconstruction training (less effective but still usable for style).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import CACHE_DIR

console = Console()


DatasetTier = Literal["i2v", "video", "image"]


BUILTIN_DATASETS: dict[str, dict] = {
    "tip-i2v": {
        "hf_dataset": "WenhaoWang/TIP-I2V",
        "tier": "i2v",
        "description": (
            "TIP-I2V (ICCV 2025) — 1.7M real user image+text prompt pairs for I2V. "
            "Each sample has an Image_Prompt (first frame) + Text_Prompt (caption). "
            "The gold standard for image-to-video LoRA training."
        ),
        "num_samples": 200,
        "max_samples": 10000,
        "download_fn": "_download_tip_i2v",
    },
    "pusa": {
        "hf_dataset": "RaphaelLiu/PusaV1_training",
        "tier": "video",
        "description": (
            "PusaV1 training set — 3,860 Wan-T2V generated video-caption pairs. "
            "Good for learning motion/style but text-to-video only (no image condition)."
        ),
        "num_samples": 200,
        "max_samples": 3860,
        "download_fn": "_download_pusa_videos",
    },
    "pexels": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "tier": "video",
        "description": (
            "Pexels stock footage — professional camera motion, smooth "
            "transitions, cinematic lighting. Downloads actual video URLs."
        ),
        "num_samples": 100,
        "max_samples": 5000,
        "filter_keywords": [
            "cinematic", "aerial", "slow motion", "timelapse", "landscape",
            "sunset", "city", "nature", "ocean", "clouds",
        ],
        "download_fn": "_download_pexels_videos",
    },
    "youtube-commons": {
        "hf_dataset": "PleIAs/YouTube-Commons",
        "tier": "video",
        "description": (
            "CC-BY YouTube corpus (transcripts + video_link). Streams parquet shards, "
            "downloads ~12s clips via yt-dlp. Use a small NUM_SAMPLES (8–32) first; many URLs fail or rate-limit."
        ),
        "num_samples": 16,
        "max_samples": 2000,
        "parquet_shards": 8,
        "download_fn": "_download_youtube_commons_videos",
    },
    "cinematic": {
        "hf_dataset": "CinematicT2vData/cinepile_captions",
        "tier": "video",
        "description": (
            "Cinematic film clips with detailed scene descriptions — camera work, "
            "lighting, composition. 3,490 video-caption pairs."
        ),
        "num_samples": 100,
        "max_samples": 3490,
        "download_fn": "_download_cinematic_videos",
    },
    "developer": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "tier": "image",
        "description": (
            "Developer/tech scenes — screens, typing, offices, code. "
            "Image-only (still frames); best for style transfer, not motion."
        ),
        "num_samples": 100,
        "max_samples": 1000,
        "filter_keywords": [
            "computer", "laptop", "coding", "typing", "office", "desk",
            "programmer", "developer", "screen", "keyboard", "technology",
            "working", "startup", "meeting",
        ],
        "download_fn": "_download_pexels_images",
    },
    "dramatic": {
        "hf_dataset": "jovianzm/Pexels-400k",
        "tier": "image",
        "description": (
            "High-contrast dramatic scenes — storms, silhouettes, moody lighting. "
            "Image-only; good for anecdote intro style."
        ),
        "num_samples": 100,
        "max_samples": 1000,
        "filter_keywords": [
            "dramatic", "dark", "moody", "storm", "rain", "night",
            "silhouette", "shadow", "intense", "frustrated", "stressed",
        ],
        "download_fn": "_download_pexels_images",
    },
}


def list_builtin_datasets() -> list[dict]:
    """Return info about available built-in datasets."""
    return [
        {
            "name": name,
            "description": cfg["description"],
            "tier": cfg["tier"],
            "samples": cfg["num_samples"],
            "hf_dataset": cfg["hf_dataset"],
        }
        for name, cfg in BUILTIN_DATASETS.items()
    ]


def download_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> Path:
    """Download a built-in dataset and prepare it for LoRA training.

    Returns the path to a directory containing:
      - For video datasets: *.mp4 files + captions.json + first-frame PNGs
      - For image datasets: *.jpg files + captions.json
    """
    if dataset_name not in BUILTIN_DATASETS:
        available = ", ".join(BUILTIN_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    cfg = BUILTIN_DATASETS[dataset_name]
    num_samples = min(max_samples or cfg["num_samples"], cfg["max_samples"])

    if output_dir is None:
        output_dir = CACHE_DIR / "datasets" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".downloaded"
    if marker.exists():
        try:
            prev_meta = json.loads(marker.read_text())
            prev_n = prev_meta.get("samples")
        except (json.JSONDecodeError, TypeError):
            prev_n = None
        if prev_n is not None and prev_n != num_samples:
            console.print(
                f"  [dim]Sample count changed ({prev_n} → {num_samples}), clearing cache...[/]"
            )
            for p in output_dir.glob("*.mp4"):
                p.unlink(missing_ok=True)
            for p in output_dir.glob("frame_*.png"):
                p.unlink(missing_ok=True)
            for p in output_dir.glob("*.jpg"):
                if not p.stem.startswith("frame_"):
                    p.unlink(missing_ok=True)
            for p in output_dir.glob("sample_*.png"):
                p.unlink(missing_ok=True)
            (output_dir / "captions.json").unlink(missing_ok=True)
        if marker.exists() and prev_n is not None and prev_n != num_samples:
            marker.unlink()
        if marker.exists():
            existing_videos = list(output_dir.glob("*.mp4"))
            existing_images = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
            total = len(existing_videos) + len(existing_images)
            if total >= max(num_samples // 4, 1):
                console.print(f"  [dim]Using cached dataset: {output_dir} ({total} files)[/]")
                return output_dir
            console.print(f"  [dim]Stale cache ({total} files), re-downloading...[/]")
            marker.unlink()

    console.print(f"[bold blue]Downloading[/] '{dataset_name}' ({cfg['tier']} tier, {num_samples} samples)")

    download_fn = globals()[cfg["download_fn"]]
    download_fn(cfg, output_dir, num_samples)

    marker.write_text(json.dumps({
        "dataset": dataset_name,
        "tier": cfg["tier"],
        "samples": num_samples,
    }))

    videos = list(output_dir.glob("*.mp4"))
    images = [f for f in output_dir.glob("*.jpg") if not f.stem.startswith("frame_")]
    images += [f for f in output_dir.glob("*.png") if not f.stem.startswith("frame_")]
    console.print(
        f"[bold green]Downloaded[/] {len(videos)} videos, {len(images)} images to {output_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Tier 0 (I2V): TIP-I2V — real image+text prompt pairs for I2V training
# ---------------------------------------------------------------------------

def _download_tip_i2v(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download image-to-video training pairs from TIP-I2V.

    Each sample has an Image_Prompt (the source image / first frame) and a
    Text_Prompt (the motion/scene description). This is exactly what the
    Wan2.1 I2V model needs: an image to condition on + a text prompt.

    We save:
      - sample_XXXX.png  (the image prompt / first frame)
      - captions.json     (text prompts keyed by filename)
      - tier marker as "i2v" so the trainer knows to use I2V conditioning
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    console.print(f"  [dim]Loading {cfg['hf_dataset']} (Eval split, streaming)...[/]")

    # TIP-I2V has 30 parquet files (27 Full + 2 Subset + 1 Eval). Resolving
    # all of them is extremely slow. Point directly at the single Eval file
    # (77MB, 10K rows) which is more than enough for 200 samples.
    try:
        ds = load_dataset(
            cfg["hf_dataset"],
            split="train",
            streaming=True,
            data_files="data/Eval-00000-of-00001.parquet",
        )
    except Exception:
        ds = load_dataset(cfg["hf_dataset"], split="Eval", streaming=True)

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading TIP-I2V pairs", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            image_prompt = row.get("Image_Prompt")
            text_prompt = row.get("Text_Prompt") or ""
            nsfw_text = row.get("Text_NSFW", 0.0) or 0.0
            nsfw_image = row.get("Image_NSFW", "SAFE") or "SAFE"

            if image_prompt is None:
                continue
            if nsfw_text > 0.3 or nsfw_image != "SAFE":
                continue
            if len(text_prompt) < 10:
                continue

            try:
                if isinstance(image_prompt, Image.Image):
                    img = image_prompt
                elif isinstance(image_prompt, dict) and "bytes" in image_prompt:
                    img = Image.open(io.BytesIO(image_prompt["bytes"]))
                else:
                    continue

                img = img.convert("RGB").resize((720, 480))
                filename = f"sample_{collected:04d}.png"
                img.save(output_dir / filename, "PNG")

                captions[filename] = text_prompt.strip()
                collected += 1
                progress.update(task, completed=collected)
            except Exception:
                continue

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Tier 1: PusaV1 — real Wan-generated videos with pre-encoded latents
# ---------------------------------------------------------------------------

def _download_pusa_videos(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download video files and captions from the PusaV1 training dataset.

    PusaV1 is a file-based HF repo (not row-based). Structure:
      train/video_000001.mp4, train/video_000002.mp4, ...
      metadata.csv  (columns: file_name, text)

    We use huggingface_hub to list and download individual mp4 files.
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub")

    import csv

    repo_id = cfg["hf_dataset"]
    api = HfApi()

    console.print(f"  [dim]Listing files in {repo_id}...[/]")

    caption_lookup: dict[str, str] = {}
    try:
        meta_path = hf_hub_download(
            repo_id=repo_id, filename="metadata.csv",
            repo_type="dataset",
        )
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("file_name") or row.get("filename") or ""
                text = row.get("text") or row.get("prompt") or row.get("caption") or ""
                if fname:
                    caption_lookup[fname] = text
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not load metadata.csv: {e}[/]")

    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    mp4_files = sorted([f for f in repo_files if f.endswith(".mp4")])

    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found in {repo_id}")

    mp4_files = mp4_files[:num_samples]
    console.print(f"  [dim]Found {len(mp4_files)} mp4 files, downloading {len(mp4_files)}...[/]")

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading PusaV1 videos", total=len(mp4_files))

        for remote_path in mp4_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id, filename=remote_path,
                    repo_type="dataset",
                )

                filename = f"pusa_{collected:04d}.mp4"
                video_path = output_dir / filename

                import shutil
                shutil.copy2(local_path, video_path)

                if not video_path.exists() or video_path.stat().st_size < 1000:
                    video_path.unlink(missing_ok=True)
                    continue

                _extract_first_frame_ffmpeg(video_path, output_dir)

                caption = caption_lookup.get(remote_path, "")
                if not caption:
                    base = Path(remote_path).stem
                    caption = caption_lookup.get(base, "")
                if not caption:
                    caption = "A smooth cinematic video clip, high quality, Wan-generated"
                captions[filename] = caption

                collected += 1
                progress.update(task, completed=collected)
            except Exception as e:
                console.print(f"  [dim]Skipping {remote_path}: {e}[/]")
                continue

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Tier 1: CinematicT2vData — film clips with rich scene descriptions
# ---------------------------------------------------------------------------

def _download_cinematic_videos(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download cinematic video clips from CinematicT2vData.

    The dataset has video_id + caption. We download the actual video if available,
    otherwise fall back to extracting YouTube clips via the video_id.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    console.print(f"  [dim]Loading {cfg['hf_dataset']}...[/]")

    ds = load_dataset(cfg["hf_dataset"], split="train", streaming=True)

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading cinematic clips", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            video_data = row.get("video")
            caption = row.get("caption") or row.get("text") or ""

            if video_data is None and not caption:
                continue

            filename = f"cinematic_{collected:04d}.mp4"
            video_path = output_dir / filename

            try:
                if video_data is not None:
                    if isinstance(video_data, dict) and "bytes" in video_data:
                        video_path.write_bytes(video_data["bytes"])
                    elif isinstance(video_data, dict) and "path" in video_data:
                        import shutil
                        shutil.copy2(video_data["path"], video_path)
                    elif isinstance(video_data, bytes):
                        video_path.write_bytes(video_data)
                    elif isinstance(video_data, str):
                        _download_file(video_data, video_path)
                    else:
                        continue
                else:
                    continue

                if not video_path.exists() or video_path.stat().st_size < 1000:
                    video_path.unlink(missing_ok=True)
                    continue

                _extract_first_frame_ffmpeg(video_path, output_dir)

                if not caption:
                    caption = "A cinematic film scene with professional cinematography"
                captions[filename] = caption

                collected += 1
                progress.update(task, completed=collected)
            except Exception:
                video_path.unlink(missing_ok=True)
                continue

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Tier 2: YouTube-Commons — CC-BY transcripts + yt-dlp short clips
# ---------------------------------------------------------------------------

def _youtube_commons_watch_url(row: dict) -> str | None:
    link = (row.get("video_link") or "").strip()
    if link and ("youtube.com" in link or "youtu.be" in link):
        return link
    vid = (row.get("video_id") or "").strip()
    if vid and len(vid) >= 8:
        return f"https://www.youtube.com/watch?v={vid}"
    return None


def _yt_dlp_section(max_seconds: int) -> str:
    """Build yt-dlp ``--download-sections`` range for the first ``max_seconds`` seconds."""
    if max_seconds <= 0:
        max_seconds = 12
    m, s = divmod(max_seconds, 60)
    return f"*0:00-{int(m)}:{s:02d}"


def _yt_dlp_download_clip(url: str, dest: Path, max_seconds: int, timeout: int) -> bool:
    """Download the first ``max_seconds`` of a video with yt-dlp. Returns True on success."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError(
            "yt-dlp not found on PATH. Install with: pip install 'repovideo[train]' or pip install yt-dlp"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    section = _yt_dlp_section(max_seconds)
    out_tmpl = str(dest.with_suffix("")) + ".%(ext)s"
    cmd = [
        ytdlp,
        "--no-playlist",
        "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "2",
        "--fragment-retries", "2",
        "-f", "bv*[height<=720][ext=mp4]+ba/b[height<=720][ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--download-sections", section,
        "-o", out_tmpl,
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        cmd_fb = [
            ytdlp,
            "--no-playlist",
            "--no-warnings",
            "--socket-timeout", "30",
            "--retries", "1",
            "-f", "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "--download-sections", section,
            "-o", out_tmpl,
            url,
        ]
        r = subprocess.run(cmd_fb, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return False
    for ext in ("mp4", "webm", "mkv"):
        p = dest.with_suffix(f".{ext}")
        if p.exists() and p.stat().st_size > 2000:
            if p != dest:
                p.rename(dest)
            return True
    base = dest.with_suffix("")
    for p in dest.parent.glob(base.name + ".*"):
        if p.suffix.lower() in (".mp4", ".webm", ".mkv") and p.stat().st_size > 2000:
            p.rename(dest)
            return True
    return False


def _caption_from_youtube_commons_row(row: dict, max_chars: int = 600) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    parts = [p for p in (title, text) if p]
    return ". ".join(parts) if parts else "A YouTube video clip, natural motion, documentary style."


def _download_youtube_commons_videos(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Stream YouTube-Commons parquet shards and download short clips with yt-dlp."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    repo = cfg["hf_dataset"]
    n_shards = min(int(cfg.get("parquet_shards", 8)), 50)
    data_files = [f"hf://datasets/{repo}/cctube_{i}.parquet" for i in range(n_shards)]

    console.print(
        f"  [dim]Streaming {n_shards} parquet shard(s) from {repo} (yt-dlp clips)...[/]"
    )

    ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    max_sec = int(os.environ.get("REPOVIDEO_YTC_CLIP_SECONDS", "12"))
    timeout = int(os.environ.get("REPOVIDEO_YTC_TIMEOUT", "180"))
    require_en = cfg.get("require_english", False)

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("YouTube-Commons + yt-dlp", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            if require_en and (row.get("original_language") or "").lower() not in ("en", "english"):
                continue

            url = _youtube_commons_watch_url(row)
            if not url:
                continue

            caption = _caption_from_youtube_commons_row(row)
            if len(caption) < 15:
                continue

            filename = f"ytc_{collected:04d}.mp4"
            video_path = output_dir / filename

            try:
                if not _yt_dlp_download_clip(url, video_path, max_sec, timeout):
                    continue
                if not video_path.exists() or video_path.stat().st_size < 3000:
                    video_path.unlink(missing_ok=True)
                    continue

                _trim_video(video_path, max_seconds=float(max_sec) + 1.0)
                _extract_first_frame_ffmpeg(video_path, output_dir)
                captions[filename] = caption
                collected += 1
                progress.update(task, completed=collected)
            except subprocess.TimeoutExpired:
                video_path.unlink(missing_ok=True)
                continue
            except Exception:
                video_path.unlink(missing_ok=True)
                continue

    if collected == 0:
        raise RuntimeError(
            "No videos downloaded from YouTube-Commons. Check yt-dlp is installed, "
            "you are not IP-blocked by YouTube, and try a smaller NUM_SAMPLES or different shards."
        )

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Tier 2: Pexels stock footage — download actual videos via URL
# ---------------------------------------------------------------------------

def _download_pexels_videos(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download actual video files from Pexels-400k, filtered by keywords."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    keywords = cfg.get("filter_keywords", [])
    console.print(f"  [dim]Loading {cfg['hf_dataset']} metadata (streaming)...[/]")
    ds = load_dataset(cfg["hf_dataset"], split="train", streaming=True)

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading Pexels videos", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            title = (row.get("title") or row.get("Title") or "").lower()
            tags = (row.get("tags") or row.get("Tags") or "").lower()
            searchable = f"{title} {tags}"

            if keywords and not any(kw in searchable for kw in keywords):
                continue

            video_url = (
                row.get("contentUrl")
                or row.get("video_url")
                or row.get("VideoUrl")
                or row.get("content_url")
            )
            if not video_url or not isinstance(video_url, str):
                continue

            filename = f"pexels_{collected:04d}.mp4"
            video_path = output_dir / filename

            try:
                _download_file(video_url, video_path, timeout=30)

                if not video_path.exists() or video_path.stat().st_size < 5000:
                    video_path.unlink(missing_ok=True)
                    continue

                _trim_video(video_path, max_seconds=4.0)
                _extract_first_frame_ffmpeg(video_path, output_dir)

                caption = title.strip() or "A stock footage clip"
                captions[filename] = f"{caption}, smooth motion, high quality, cinematic"

                collected += 1
                progress.update(task, completed=collected)
            except Exception:
                video_path.unlink(missing_ok=True)
                continue

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Tier 3: Image-only fallback — Pexels thumbnails (no motion learning)
# ---------------------------------------------------------------------------

def _download_pexels_images(cfg: dict, output_dir: Path, num_samples: int) -> None:
    """Download thumbnail images from Pexels-400k, filtered by keywords.

    Image-only mode: useful for style transfer LoRA training but won't teach
    the model anything about motion or temporal coherence.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the 'datasets' library: pip install datasets")

    keywords = cfg.get("filter_keywords", [])
    console.print(f"  [dim]Loading {cfg['hf_dataset']} metadata (streaming)...[/]")
    ds = load_dataset(cfg["hf_dataset"], split="train", streaming=True)

    collected = 0
    captions: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading '{cfg.get('filter_keywords', [''])[0]}' images", total=num_samples)

        for row in ds:
            if collected >= num_samples:
                break

            title = (row.get("title") or row.get("Title") or "").lower()
            tags = (row.get("tags") or row.get("Tags") or "").lower()
            searchable = f"{title} {tags}"

            if keywords and not any(kw in searchable for kw in keywords):
                continue

            thumbnail = row.get("thumbnail") or row.get("Thumbnail")
            if thumbnail is None:
                continue

            try:
                if isinstance(thumbnail, str):
                    img = _download_image_from_url(thumbnail)
                elif isinstance(thumbnail, Image.Image):
                    img = thumbnail
                elif isinstance(thumbnail, dict) and "bytes" in thumbnail:
                    img = Image.open(io.BytesIO(thumbnail["bytes"]))
                else:
                    continue

                img = img.convert("RGB").resize((720, 480))
                filename = f"sample_{collected:04d}.jpg"
                img.save(output_dir / filename, "JPEG", quality=90)

                caption = title.strip() or "a scene, smooth motion, cinematic"
                captions[filename] = f"{caption}, smooth motion, high quality, cinematic"
                collected += 1
                progress.update(task, completed=collected)
            except Exception:
                continue

    (output_dir / "captions.json").write_text(json.dumps(captions, indent=2))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, timeout: int = 15) -> None:
    """Download a file from URL to disk."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "repovideo/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def _download_image_from_url(url: str) -> Image.Image:
    """Download a single image from URL."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read()))


def _extract_first_frame_ffmpeg(video_path: Path, output_dir: Path) -> Path | None:
    """Extract the first frame of a video as a PNG for I2V training pairs."""
    frame_path = output_dir / f"frame_{video_path.stem}.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "scale=720:480",
        "-vframes", "1",
        str(frame_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and frame_path.exists():
        return frame_path
    return None


def _trim_video(video_path: Path, max_seconds: float = 4.0) -> None:
    """Trim a video to max_seconds from the start (in-place)."""
    trimmed = video_path.with_suffix(".trimmed.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-t", str(max_seconds),
        "-c:v", "libx264", "-preset", "ultrafast", "-an",
        str(trimmed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and trimmed.exists() and trimmed.stat().st_size > 100:
        trimmed.replace(video_path)
    else:
        trimmed.unlink(missing_ok=True)


def get_dataset_tier(dataset_name: str) -> DatasetTier:
    """Return whether a built-in dataset provides video or image-only data."""
    if dataset_name not in BUILTIN_DATASETS:
        return "image"
    return BUILTIN_DATASETS[dataset_name]["tier"]


def prepare_builtin_dataset_for_training(dataset_name: str) -> tuple[Path, DatasetTier]:
    """Download dataset and return (directory, tier) ready for LoRA training."""
    output_dir = download_dataset(dataset_name)
    tier = get_dataset_tier(dataset_name)
    return output_dir, tier
