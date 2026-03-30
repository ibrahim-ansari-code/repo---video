"""Stage 4a: Playwright browser recorder for web app demos."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright, Page, BrowserContext
from rich.console import Console

from src.analyzer import RepoManifest
from src.config import DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_FPS
from src.recorder.script_generator import DemoAction, DemoScript, generate_web_demo_script

console = Console()


async def record_web_demo(
    manifest: RepoManifest,
    host_port: int,
    output_path: Path,
    duration: int = 60,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
) -> Path:
    """Record a browser demo of a web app running on the given port."""
    script = generate_web_demo_script(manifest, host_port)
    video_dir = Path(tempfile.mkdtemp(prefix="repovideo_browser_"))

    console.print(f"[bold blue]Recording[/] browser demo → {output_path}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        context.set_default_timeout(10_000)
        page = await context.new_page()

        try:
            await _execute_demo_script(page, script, duration)
        except Exception as e:
            console.print(f"[yellow]Warning during recording:[/] {e}")
        finally:
            await page.close()
            await context.close()
            await browser.close()

    video_files = list(video_dir.glob("*.webm"))
    if not video_files:
        raise RuntimeError("No video file was produced by Playwright")

    recorded = video_files[0]
    _convert_webm_to_mp4(recorded, output_path)
    console.print(f"[bold green]Recorded[/] browser demo: {output_path}")
    return output_path


async def _execute_demo_script(page: Page, script: DemoScript, max_duration: int) -> None:
    """Run through each action in the demo script."""
    import time
    start = time.time()

    for action in script.actions:
        if time.time() - start > max_duration:
            console.print("[yellow]Reached max duration, stopping demo[/]")
            break

        try:
            await _perform_action(page, action)
        except Exception as e:
            console.print(f"[dim]Skipping action '{action.description}': {e}[/]")

        await asyncio.sleep(action.wait_ms / 1000)


async def _perform_action(page: Page, action: DemoAction) -> None:
    if action.action == "navigate":
        await page.goto(action.value, wait_until="domcontentloaded")

    elif action.action == "click":
        locator = page.locator(action.selector).first
        if await locator.count() > 0:
            await locator.scroll_into_view_if_needed()
            await locator.click()

    elif action.action == "type":
        locator = page.locator(action.selector).first
        if await locator.count() > 0:
            await locator.scroll_into_view_if_needed()
            await locator.fill(action.value)

    elif action.action == "scroll":
        if action.value == "bottom":
            await page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
        elif action.value == "top":
            await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        else:
            await page.evaluate(f"window.scrollBy({{top: {action.value}, behavior: 'smooth'}})")

    elif action.action == "wait":
        await asyncio.sleep(action.wait_ms / 1000)

    elif action.action == "screenshot":
        pass  # video is being recorded continuously


def _convert_webm_to_mp4(input_path: Path, output_path: Path) -> None:
    """Convert Playwright's WebM output to MP4 using ffmpeg."""
    import subprocess

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


def run_web_recording(
    manifest: RepoManifest,
    host_port: int,
    output_path: Path,
    duration: int = 60,
) -> Path:
    """Synchronous wrapper around the async recording function."""
    return asyncio.run(record_web_demo(manifest, host_port, output_path, duration))
