"""Stage 4a: Playwright browser recorder for web app demos."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from playwright.async_api import Page, async_playwright
from rich.console import Console

from src.analyzer import RepoManifest
from src.config import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_FPS
from src.recorder.script_generator import DemoAction, DemoScript, generate_web_demo_script

console = Console()


async def _wait_spa_ready(page: Page) -> None:
    """Wait for Vite/React (or similar) to paint real content — not just a blank shell."""
    await asyncio.sleep(0.45)
    for sel in ("#root", "#app", "[data-reactroot]", "main"):
        try:
            await page.locator(sel).first.wait_for(state="attached", timeout=15_000)
            break
        except Exception:
            continue
    try:
        await page.wait_for_function(
            """() => {
                const el = document.querySelector('#root')
                    || document.querySelector('#app')
                    || document.body;
                if (!el) return false;
                const t = (el.innerText || '').trim();
                return t.length > 2;
            }""",
            timeout=30_000,
        )
    except Exception:
        pass
    await asyncio.sleep(1.0)


async def _explore_clickables(page: Page, max_actions: int) -> None:
    """Click real controls in the live app (buttons, links, test ids), then try text inputs."""
    if max_actions <= 0:
        return
    done = 0

    click_selectors = [
        "button:not([disabled]):visible",
        "[role='button']:not([aria-disabled='true']):visible",
        "a[href^='/']:visible",
        "a[href^='#']:visible",
        "[data-testid]:visible",
    ]
    seen_text: set[str] = set()

    for sel in click_selectors:
        if done >= max_actions:
            break
        loc = page.locator(sel)
        try:
            n = await loc.count()
        except Exception:
            continue
        for i in range(min(n, 5)):
            if done >= max_actions:
                break
            el = loc.nth(i)
            try:
                if not await el.is_visible():
                    continue
                href = await el.get_attribute("href")
                if href in ("#", ""):
                    continue
                txt = ((await el.inner_text()) or "").strip()[:120]
                if txt and txt in seen_text:
                    continue
                if txt:
                    seen_text.add(txt)
                await el.scroll_into_view_if_needed()
                await el.click(timeout=8_000)
                done += 1
                await asyncio.sleep(2.0)
            except Exception as e:
                console.print(f"[dim]explore skip ({sel}): {e}[/]")

    try:
        inputs = page.locator("input:not([type='hidden']):visible, textarea:visible")
        n_in = await inputs.count()
        for i in range(min(n_in, 3)):
            if done >= max_actions:
                break
            el = inputs.nth(i)
            try:
                itype = (await el.get_attribute("type") or "text").lower()
                if itype in ("submit", "button", "checkbox", "radio", "file", "image"):
                    continue
                await el.scroll_into_view_if_needed()
                await el.click(timeout=5_000)
                name_attr = (await el.get_attribute("name") or "").lower()
                fill_val = (
                    "demo@example.com"
                    if itype == "email" or "email" in name_attr
                    else "Demo"
                )
                await el.fill(fill_val, timeout=4_000)
                done += 1
                await asyncio.sleep(1.2)
            except Exception:
                continue
    except Exception:
        pass


async def record_web_demo(
    manifest: RepoManifest,
    app_url: str,
    output_path: Path,
    duration: int = 60,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
) -> Path:
    """Record a browser demo of a web app at the given URL."""
    script = generate_web_demo_script(manifest, app_url=app_url)
    video_dir = Path(tempfile.mkdtemp(prefix="repovideo_browser_"))

    console.print(f"[bold blue]Recording[/] browser demo → {output_path}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
        )
        context.set_default_timeout(45_000)
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

        if action.wait_ms > 0:
            await asyncio.sleep(action.wait_ms / 1000)


async def _perform_action(page: Page, action: DemoAction) -> None:
    if action.action == "navigate":
        await page.goto(action.value, wait_until="load", timeout=120_000)
        await _wait_spa_ready(page)

    elif action.action == "explore_ui":
        try:
            n = int((action.value or "8").strip())
        except ValueError:
            n = 8
        await _explore_clickables(page, max(1, min(n, 24)))

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
        pass


def _convert_webm_to_mp4(input_path: Path, output_path: Path) -> None:
    """Convert Playwright's WebM output to MP4 using ffmpeg."""
    import subprocess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


def run_web_recording(
    manifest: RepoManifest,
    app_url: str,
    output_path: Path,
    duration: int = 60,
) -> Path:
    """Synchronous wrapper around the async recording function."""
    return asyncio.run(record_web_demo(manifest, app_url, output_path, duration))
