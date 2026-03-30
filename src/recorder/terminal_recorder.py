"""Stage 4b: Terminal recorder for CLI tool demos.

Records CLI sessions using subprocess-based pseudo-terminal recording,
then converts to MP4 by rendering in a styled HTML terminal and capturing
with Playwright.
"""

from __future__ import annotations

import asyncio
import json
import os
import pty
import select
import subprocess
import tempfile
import time
from pathlib import Path

from rich.console import Console

from src.analyzer import RepoManifest
from src.config import DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
from src.recorder.script_generator import generate_cli_demo_script

console = Console()

ASCIINEMA_PLAYER_CDN = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0"

TERMINAL_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="{player_cdn}/dist/bundle/asciinema-player.css" />
<style>
  body {{
    margin: 0;
    padding: 40px;
    background: #1a1b26;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
  }}
  #player {{
    width: {width}px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }}
  .title-bar {{
    background: #24283b;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .dot-red {{ background: #f7768e; }}
  .dot-yellow {{ background: #e0af68; }}
  .dot-green {{ background: #9ece6a; }}
  .title-text {{
    color: #7aa2f7;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    margin-left: 12px;
  }}
</style>
</head>
<body>
  <div id="player">
    <div class="title-bar">
      <div class="dot dot-red"></div>
      <div class="dot dot-yellow"></div>
      <div class="dot dot-green"></div>
      <span class="title-text">{title}</span>
    </div>
    <div id="terminal"></div>
  </div>
  <script src="{player_cdn}/dist/bundle/asciinema-player.min.js"></script>
  <script>
    const castData = {cast_json};
    const blob = new Blob([castData], {{ type: 'text/plain' }});
    const url = URL.createObjectURL(blob);
    AsciinemaPlayer.create(url, document.getElementById('terminal'), {{
      autoPlay: true,
      speed: 1,
      theme: 'dracula',
      fit: 'width',
      fontSize: 'big',
      rows: 30,
      cols: 100,
    }});
  </script>
</body>
</html>"""


def record_cli_demo(
    manifest: RepoManifest,
    sandbox_exec_fn,
    output_path: Path,
    duration: int = 60,
) -> Path:
    """Record a CLI demo by executing commands and capturing output as a cast file,
    then rendering to MP4 via Playwright."""
    commands = generate_cli_demo_script(manifest)

    if not commands:
        commands = [f"echo 'Demo of {manifest.name}'", f"{manifest.run_command} --help"]

    console.print(f"[bold blue]Recording[/] terminal demo ({len(commands)} commands)")

    cast_path = Path(tempfile.mktemp(suffix=".cast", prefix="repovideo_"))
    _record_cast(commands, sandbox_exec_fn, cast_path, duration)

    _cast_to_mp4(cast_path, output_path, manifest.name)
    console.print(f"[bold green]Recorded[/] terminal demo: {output_path}")
    return output_path


def _record_cast(
    commands: list[str],
    sandbox_exec_fn,
    cast_path: Path,
    max_duration: int,
) -> None:
    """Generate an asciicast v2 file from command executions."""
    events: list[tuple[float, str, str]] = []
    start_time = time.time()

    header = {
        "version": 2,
        "width": 100,
        "height": 30,
        "timestamp": int(start_time),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }

    for cmd in commands:
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            break

        prompt = f"\x1b[1;32m$\x1b[0m {cmd}\r\n"
        events.append((elapsed, "o", prompt))

        typing_delay = 0.05
        for char in cmd:
            elapsed = time.time() - start_time
            events.append((elapsed, "o", ""))
            time.sleep(0.01)

        elapsed = time.time() - start_time
        events.append((elapsed + 0.3, "o", ""))

        try:
            exit_code, output = sandbox_exec_fn(cmd)
            output_lines = output.replace("\n", "\r\n")
            elapsed = time.time() - start_time
            events.append((elapsed, "o", output_lines + "\r\n"))
        except Exception as e:
            elapsed = time.time() - start_time
            events.append((elapsed, "o", f"\x1b[31mError: {e}\x1b[0m\r\n"))

        time.sleep(0.5)

    with open(cast_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for ts, event_type, data in events:
            f.write(json.dumps([ts, event_type, data]) + "\n")


def _cast_to_mp4(cast_path: Path, output_path: Path, title: str) -> None:
    """Render the cast file in a browser-based terminal player and capture as MP4."""
    cast_content = cast_path.read_text()

    html_content = TERMINAL_HTML_TEMPLATE.format(
        player_cdn=ASCIINEMA_PLAYER_CDN,
        width=DEFAULT_VIDEO_WIDTH - 80,
        title=title,
        cast_json=json.dumps(cast_content),
    )

    html_path = Path(tempfile.mktemp(suffix=".html", prefix="repovideo_term_"))
    html_path.write_text(html_content)

    asyncio.run(_capture_html_as_video(html_path, output_path))


async def _capture_html_as_video(html_path: Path, output_path: Path) -> None:
    """Use Playwright to open the HTML terminal player and record the video."""
    from playwright.async_api import async_playwright

    video_dir = Path(tempfile.mkdtemp(prefix="repovideo_termvid_"))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": DEFAULT_VIDEO_WIDTH, "height": DEFAULT_VIDEO_HEIGHT},
            record_video_dir=str(video_dir),
            record_video_size={"width": DEFAULT_VIDEO_WIDTH, "height": DEFAULT_VIDEO_HEIGHT},
        )
        page = await context.new_page()
        await page.goto(f"file://{html_path}", wait_until="load")

        await asyncio.sleep(2)

        cast_duration = await _get_cast_duration(page)
        await asyncio.sleep(cast_duration + 2)

        await page.close()
        await context.close()
        await browser.close()

    video_files = list(video_dir.glob("*.webm"))
    if not video_files:
        raise RuntimeError("No video captured from terminal player")

    _convert_webm_to_mp4(video_files[0], output_path)


async def _get_cast_duration(page) -> float:
    """Try to extract the duration from the cast data loaded in the player."""
    try:
        duration = await page.evaluate("""
            () => {
                const events = document.querySelector('#terminal')?.__player?.getDuration?.();
                return events || 10;
            }
        """)
        return min(float(duration), 120)
    except Exception:
        return 15


def _convert_webm_to_mp4(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
