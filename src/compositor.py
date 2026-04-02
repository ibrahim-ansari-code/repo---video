"""Stage 5: Video Compositor — stitch title cards, anecdote, transitions, demo, and outro into a final MP4."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from src.config import DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_FPS, ASSETS_DIR

console = Console()

_FFMPEG_DRAWTEXT: bool | None = None


def _ffmpeg_has_drawtext() -> bool:
    """Homebrew ffmpeg often ships without libfreetype, so drawtext is missing."""
    global _FFMPEG_DRAWTEXT
    if _FFMPEG_DRAWTEXT is not None:
        return _FFMPEG_DRAWTEXT
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.1",
            "-vf",
            "drawtext=text='x':fontsize=12:fontcolor=white:x=1:y=1",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    err = (r.stderr or "") + (r.stdout or "")
    _FFMPEG_DRAWTEXT = r.returncode == 0 and "No such filter" not in err
    return _FFMPEG_DRAWTEXT


def _load_truetype_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _multiline_text_size(font: ImageFont.ImageFont, text: str, spacing: int = 4) -> tuple[int, int]:
    lines = text.split("\n")
    max_w = 0
    total_h = 0
    for line in lines:
        if not line:
            total_h += spacing
            continue
        l, t, r, b = font.getbbox(line)
        max_w = max(max_w, r - l)
        total_h += b - t + spacing
    return max_w, max(0, total_h - spacing)


def _video_stream_dimensions(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = r.stdout.strip().split("x")
    return int(w), int(h)


def _png_to_video_segment(png_path: Path, out_mp4: Path, duration: float, fps: int, desc: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(png_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(duration),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    _run_ffmpeg(cmd, desc)


def _generate_title_card_pillow(
    output_path: Path,
    name: str,
    description: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    img = Image.new("RGB", (width, height), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    desc = (description or "").strip()
    name_size = min(72, max(28, width // max(len(name), 8)))
    font_name = _load_truetype_font(name_size)

    if not desc:
        draw.text((width // 2, height // 2), name, font=font_name, fill=(255, 255, 255), anchor="mm")
    else:
        desc_size = min(36, max(18, width // 48))
        font_desc = _load_truetype_font(desc_size)
        draw.text((width // 2, height // 2 - 55), name, font=font_name, fill=(255, 255, 255), anchor="mm")
        lines = textwrap.wrap(desc[:400], width=max(24, width // 28))[:5]
        line_gap = font_desc.getbbox("Ay")[3] - font_desc.getbbox("Ay")[1] + 10
        y = height // 2 + 25
        for line in lines:
            draw.text((width // 2, y), line, font=font_desc, fill=(204, 204, 204), anchor="mm")
            y += line_gap

    png_path = output_path.with_suffix(".title.png")
    img.save(png_path)
    try:
        _png_to_video_segment(png_path, output_path, TITLE_DURATION, fps, "title card (pillow)")
    finally:
        png_path.unlink(missing_ok=True)


def _generate_outro_card_pillow(
    output_path: Path,
    name: str,
    repo_url: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    img = Image.new("RGB", (width, height), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    font_lg = _load_truetype_font(min(48, width // 28))
    font_sm = _load_truetype_font(min(28, width // 42))
    line1 = f"Star {name} on GitHub"
    draw.text((width // 2, height // 2 - 45), line1, font=font_lg, fill=(255, 255, 255), anchor="mm")
    draw.text((width // 2, height // 2 + 35), repo_url, font=font_sm, fill=(96, 165, 250), anchor="mm")

    png_path = output_path.with_suffix(".outro.png")
    img.save(png_path)
    try:
        _png_to_video_segment(png_path, output_path, OUTRO_DURATION, fps, "outro card (pillow)")
    finally:
        png_path.unlink(missing_ok=True)


def _add_text_overlay_pillow(
    input_path: Path,
    output_path: Path,
    text: str,
    position: str = "bottom",
) -> None:
    w, h = _video_stream_dimensions(input_path)
    bar_h = max(90, min(200, h // 5))
    bar = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(bar)
    d.rounded_rectangle((0, 0, w, bar_h), radius=14, fill=(0, 0, 0, 200))
    font = _load_truetype_font(min(36, max(20, w // 36)))
    wrapped = textwrap.fill(text, width=max(16, w // 22))
    tw, th = _multiline_text_size(font, wrapped)
    tx = max(16, (w - tw) // 2)
    ty = (bar_h - th) // 2 if position == "bottom" else max(8, (bar_h - th) // 2)
    d.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255), spacing=6)

    overlay_png = Path(tempfile.mktemp(suffix=".overlay.png"))
    try:
        bar.save(overlay_png)
        y_expr = "main_h-overlay_h" if position == "bottom" else "(main_h-overlay_h)/2"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(overlay_png),
            "-filter_complex",
            f"[0:v][1:v]overlay=0:{y_expr}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        _run_ffmpeg(cmd, "text overlay (pillow)")
    finally:
        overlay_png.unlink(missing_ok=True)

TITLE_DURATION = 3.0
OUTRO_DURATION = 3.0
CROSSFADE_DURATION = 1.0


def subtitle_for_title_card(package_description: str) -> str:
    """One-line title subtitle from package.json only — never README install steps."""
    s = (package_description or "").strip()
    if len(s) < 12 or len(s) > 140:
        return ""
    low = s.lower()
    for bad in (
        "install depend",
        "npm install",
        "yarn ",
        "pnpm ",
        "git clone",
        "build ",
        "compile",
        "```",
    ):
        if bad in low:
            return ""
    return s


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
    if not _ffmpeg_has_drawtext():
        _generate_title_card_pillow(output_path, name, description, width, height, fps)
        return

    safe_name = _escape_ffmpeg_text(name)
    safe_desc = _escape_ffmpeg_text(description[:80] if description else "")

    font_size_name = min(72, width // (len(name) + 1))
    font_size_desc = min(36, width // max(len(safe_desc), 1))

    name_y = "(h-text_h)/2" if not safe_desc else "(h-text_h)/2-40"
    drawtext_name = (
        f"drawtext=text='{safe_name}'"
        f":fontsize={font_size_name}:fontcolor=white"
        f":x=(w-text_w)/2:y={name_y}"
        f":alpha='if(lt(t,0.5),t/0.5,if(gt(t,{TITLE_DURATION - 0.5}),(({TITLE_DURATION}-t)/0.5),1))'"
    )

    drawtext_desc = (
        f"drawtext=text='{safe_desc}'"
        f":fontsize={font_size_desc}:fontcolor=0xCCCCCC"
        f":x=(w-text_w)/2:y=(h/2)+30"
        f":alpha='if(lt(t,0.8),t/0.8,if(gt(t,{TITLE_DURATION - 0.5}),(({TITLE_DURATION}-t)/0.5),1))'"
    )

    vf = drawtext_name if not safe_desc else f"{drawtext_name},{drawtext_desc}"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x111827:s={width}x{height}:d={TITLE_DURATION}:r={fps}",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(TITLE_DURATION),
        "-vf", vf,
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
    if not _ffmpeg_has_drawtext():
        _generate_outro_card_pillow(output_path, name, repo_url, width, height, fps)
        return

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
    if not _ffmpeg_has_drawtext():
        _add_text_overlay_pillow(input_path, output_path, text, position=position)
        return

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
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(a)],
        capture_output=True, text=True,
    )
    try:
        a_duration = float(probe.stdout.strip())
    except (ValueError, AttributeError):
        _simple_concat([a, b], output_path)
        return

    offset = max(0, a_duration - duration)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(a),
        "-i", str(b),
        "-filter_complex", (
            f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset={offset:.3f}[v];"
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

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(input_path)],
        capture_output=True, text=True,
    )
    try:
        vid_duration = float(probe.stdout.strip())
    except (ValueError, AttributeError):
        import shutil
        shutil.copy2(input_path, output_path)
        return

    fade_out_start = max(0, vid_duration - 2)
    music_path = music_files[0]
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", str(music_path),
        "-filter_complex", (
            f"[1:a]volume=0.08,afade=t=in:d=2,afade=t=out:st={fade_out_start:.2f}:d=2[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first[aout]"
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
