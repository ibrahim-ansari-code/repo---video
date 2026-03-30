"""Stage 5: Video Compositor — stitch title cards, anecdote, transitions, demo, and outro into a final MP4."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from src.config import DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_FPS, ASSETS_DIR

console = Console()

TITLE_DURATION = 3.0
OUTRO_DURATION = 3.0
CROSSFADE_DURATION = 1.0


def composite_video(
    demo_path: Path,
    output_path: Path,
    project_name: str,
    project_description: str,
    repo_url: str,
    anecdote_path: Path | None = None,
    overlay_text: str = "Sound familiar?",
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> Path:
    """Build the final video from all segments."""
    console.print("[bold blue]Compositing[/] final video")

    work_dir = Path(tempfile.mkdtemp(prefix="repovideo_comp_"))
    segments: list[Path] = []

    title_path = work_dir / "title.mp4"
    _generate_title_card(title_path, project_name, project_description, width, height, fps)
    segments.append(title_path)

    if anecdote_path and anecdote_path.exists():
        scaled_anecdote = work_dir / "anecdote_scaled.mp4"
        _scale_video(anecdote_path, scaled_anecdote, width, height, fps)

        if overlay_text:
            overlaid = work_dir / "anecdote_overlaid.mp4"
            _add_text_overlay(scaled_anecdote, overlaid, overlay_text, position="bottom")
            segments.append(overlaid)
        else:
            segments.append(scaled_anecdote)

    scaled_demo = work_dir / "demo_scaled.mp4"
    _scale_video(demo_path, scaled_demo, width, height, fps)
    segments.append(scaled_demo)

    outro_path = work_dir / "outro.mp4"
    _generate_outro_card(outro_path, project_name, repo_url, width, height, fps)
    segments.append(outro_path)

    if len(segments) > 1:
        joined = work_dir / "joined.mp4"
        _concat_with_crossfades(segments, joined, CROSSFADE_DURATION)
    else:
        joined = segments[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    final = work_dir / "final.mp4"
    _add_background_music(joined, final)

    import shutil
    shutil.copy2(final, output_path)

    console.print(f"[bold green]Final video:[/] {output_path}")
    return output_path


def _generate_title_card(
    output_path: Path,
    name: str,
    description: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Generate a title card video with the project name and description."""
    safe_name = _escape_ffmpeg_text(name)
    safe_desc = _escape_ffmpeg_text(description[:80] if description else "")

    font_size_name = min(72, width // (len(name) + 1))
    font_size_desc = min(36, width // max(len(safe_desc), 1))

    drawtext_name = (
        f"drawtext=text='{safe_name}'"
        f":fontsize={font_size_name}:fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2-40"
        f":alpha='if(lt(t,0.5),t/0.5,if(gt(t,{TITLE_DURATION - 0.5}),(({TITLE_DURATION}-t)/0.5),1))'"
    )

    drawtext_desc = (
        f"drawtext=text='{safe_desc}'"
        f":fontsize={font_size_desc}:fontcolor=0xCCCCCC"
        f":x=(w-text_w)/2:y=(h/2)+30"
        f":alpha='if(lt(t,0.8),t/0.8,if(gt(t,{TITLE_DURATION - 0.5}),(({TITLE_DURATION}-t)/0.5),1))'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x111827:s={width}x{height}:d={TITLE_DURATION}:r={fps}",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(TITLE_DURATION),
        "-vf", f"{drawtext_name},{drawtext_desc}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-shortest",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "title card")


def _generate_outro_card(
    output_path: Path,
    name: str,
    repo_url: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Generate an outro card with the repo URL."""
    safe_name = _escape_ffmpeg_text(name)
    safe_url = _escape_ffmpeg_text(repo_url)

    drawtext_star = (
        f"drawtext=text='\\⭐ Star {safe_name} on GitHub'"
        f":fontsize=48:fontcolor=white"
        f":x=(w-text_w)/2:y=(h/2)-40"
        f":alpha='if(lt(t,0.5),t/0.5,1)'"
    )

    drawtext_url = (
        f"drawtext=text='{safe_url}'"
        f":fontsize=28:fontcolor=0x60A5FA"
        f":x=(w-text_w)/2:y=(h/2)+30"
        f":alpha='if(lt(t,0.8),t/0.8,1)'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x111827:s={width}x{height}:d={OUTRO_DURATION}:r={fps}",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(OUTRO_DURATION),
        "-vf", f"{drawtext_star},{drawtext_url}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-shortest",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "outro card")


def _scale_video(input_path: Path, output_path: Path, width: int, height: int, fps: int) -> None:
    """Scale/pad a video to the target resolution and framerate."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x111827,"
            f"fps={fps}"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "scale video")


def _add_text_overlay(
    input_path: Path,
    output_path: Path,
    text: str,
    position: str = "bottom",
) -> None:
    """Add a text overlay to an existing video."""
    safe_text = _escape_ffmpeg_text(text)
    y_expr = "h-th-60" if position == "bottom" else "(h-th)/2"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            f"drawtext=text='{safe_text}'"
            f":fontsize=36:fontcolor=white"
            f":x=(w-text_w)/2:y={y_expr}"
            f":box=1:boxcolor=black@0.5:boxborderw=10"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "text overlay")


def _concat_with_crossfades(
    segments: list[Path],
    output_path: Path,
    fade_duration: float,
) -> None:
    """Concatenate video segments with crossfade transitions.

    Falls back to simple concat if crossfade filter is too complex
    for the number of segments.
    """
    if len(segments) <= 1:
        import shutil
        shutil.copy2(segments[0], output_path)
        return

    if len(segments) == 2:
        _crossfade_two(segments[0], segments[1], output_path, fade_duration)
        return

    _simple_concat(segments, output_path)


def _crossfade_two(a: Path, b: Path, output_path: Path, duration: float) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(a),
        "-i", str(b),
        "-filter_complex", (
            f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset="
            f"$(ffprobe -v error -show_entries format=duration -of csv=p=0 {a})-{duration}[v];"
            f"[0:a][1:a]acrossfade=d={duration}[a]"
        ),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _simple_concat([a, b], output_path)


def _simple_concat(segments: list[Path], output_path: Path) -> None:
    """Fallback: simple concatenation without transitions."""
    concat_file = Path(tempfile.mktemp(suffix=".txt"))
    with open(concat_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "concat segments")


def _add_background_music(input_path: Path, output_path: Path) -> None:
    """Add low-volume background music if available, otherwise just copy."""
    music_dir = ASSETS_DIR / "music"
    music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav")) if music_dir.exists() else []

    if not music_files:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    music_path = music_files[0]
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", str(music_path),
        "-filter_complex", (
            "[1:a]volume=0.08,afade=t=in:d=2,afade=t=out:st=END-2:d=2[bg];"
            "[0:a][bg]amix=inputs=2:duration=first[aout]"
        ),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy2(input_path, output_path)


def _escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for ffmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def _run_ffmpeg(cmd: list[str], description: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[yellow]ffmpeg warning ({description}):[/] {result.stderr[:200]}")
        raise RuntimeError(f"ffmpeg failed for {description}: {result.stderr[:500]}")
