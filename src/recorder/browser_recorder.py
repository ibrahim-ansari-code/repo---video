"""Stage 4a: Playwright browser recorder for web app demos."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright
from rich.console import Console

from src.analyzer import RepoManifest
from src.config import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_FPS
from src.recorder.script_generator import DemoAction, DemoScript, generate_web_demo_script

console = Console()


def _in_app_href(href: str | None, page_url: str) -> bool:
    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
        return False
    if href.startswith("/"):
        return True
    try:
        u = urlparse(href)
        base = urlparse(page_url)
        if u.scheme in ("http", "https") and u.netloc and u.netloc != base.netloc:
            return False
        return bool(u.path or href)
    except Exception:
        return False


async def _wait_spa_ready(page: Page) -> None:
    """Wait for Vite/React bundle to mount real UI (load fires before ES modules run)."""
    await page.wait_for_selector("#root", state="attached", timeout=120_000)
    # `load` is not enough for <script type="module"> — wait for actual chrome.
    await page.wait_for_selector(
        "#root button, #root a[href], nav a[href], #root [role='button'], #root div.cursor-pointer",
        timeout=120_000,
        state="visible",
    )
    await asyncio.sleep(0.6)


async def _dismiss_blocking_overlays(page: Page) -> None:
    """Close common full-screen modals (welcome START, OK, cookie banners, etc.)."""
    # Headless Playwright often misses styled buttons — use DOM clicks + loose text match.
    for _ in range(3):
        try:
            label = await page.evaluate("""() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const candidates = [
                ...document.querySelectorAll('button'),
                ...document.querySelectorAll('[role="button"]'),
              ];
              for (const b of candidates) {
                const t = norm(b.innerText).toUpperCase();
                if (!t) continue;
                if (t === 'START' || t === 'OK' || t === 'CLOSE' || t.startsWith('GOT IT')
                    || /^START\\b/.test(t) || t.includes('START') && t.length < 24) {
                  (b).click();
                  return norm(b.innerText) || 'START';
                }
              }
              return '';
            }""")
            if label:
                await asyncio.sleep(1.2)
                console.print(f"  [dim]Dismissed modal (DOM):[/] {label!r}")
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)

    for click_fn in (
        lambda: page.get_by_text("START", exact=True).click(timeout=6_000),
        lambda: page.get_by_role("button", name=re.compile(r"start", re.I)).first.click(timeout=6_000),
        lambda: page.locator("button").filter(has_text=re.compile(r"^\s*START\s*$", re.I)).first.click(
            timeout=6_000
        ),
    ):
        try:
            await click_fn()
            await asyncio.sleep(1.0)
            console.print("  [dim]Dismissed modal via START[/]")
            return
        except Exception:
            continue

    name_patterns = (
        r"^\s*START\s*$",
        r"^\s*Start\s*$",
        r"^\s*OK\s*$",
        r"^\s*Got it\b",
        r"^\s*Close\s*$",
        r"^\s*Continue\s*$",
        r"^\s*Dismiss\s*$",
        r"^\s*Accept\s*$",
        r"^\s*I understand\b",
    )
    for pat in name_patterns:
        try:
            btn = page.get_by_role("button", name=re.compile(pat, re.I))
            n = await btn.count()
            if n == 0:
                continue
            first = btn.first
            if await first.is_visible():
                await first.click(timeout=6_000)
                await asyncio.sleep(1.0)
                console.print("  [dim]Dismissed modal via button[/]")
                return
        except Exception:
            continue

    try:
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.55)
    except Exception:
        pass

    # Backdrop click (corner) — e.g. WelcomePopup closes on overlay tap.
    try:
        vp = page.viewport_size
        if vp:
            await page.mouse.click(12, 12)
            await page.mouse.click(vp["width"] - 12, 12)
            await asyncio.sleep(0.75)
            console.print("  [dim]Sent backdrop clicks[/]")
    except Exception:
        pass

    # If clicks never hit the React handler, remove obvious full-screen fixed layers (demo-only).
    try:
        stripped = await page.evaluate("""() => {
          let n = 0;
          const vw = window.innerWidth, vh = window.innerHeight;
          document.querySelectorAll('div').forEach((d) => {
            const st = getComputedStyle(d);
            if (st.position !== 'fixed') return;
            const r = d.getBoundingClientRect();
            const z = parseInt(String(st.zIndex || '0'), 10);
            const big = r.width >= vw * 0.65 && r.height >= vh * 0.65;
            const huge = r.width >= vw * 0.88 && r.height >= vh * 0.88;
            const highZ = Number.isFinite(z) && z >= 10;
            if (big && (highZ || huge)) {
              d.remove();
              n++;
            }
          });
          return n;
        }""")
        if stripped:
            await asyncio.sleep(0.9)
            console.print(
                f"  [dim]Removed {stripped} full-screen fixed overlay(s) so nav is reachable[/]"
            )
    except Exception:
        pass


async def _tour_navbar(page: Page, dwell_ms: int) -> None:
    """Click each unique in-app nav link once (SPA / React Router)."""
    dwell = max(1.8, dwell_ms / 1000.0)
    seen: set[str] = set()
    await asyncio.sleep(1.5)

    async def click_href_links() -> bool:
        page_url = page.url
        links = page.locator(
            "nav a[href], header a[href], [role='navigation'] a[href], "
            "a[href][class*='underline']"
        )
        try:
            n = await links.count()
        except Exception:
            return False
        if n == 0:
            return False
        for i in range(n):
            link = links.nth(i)
            try:
                href = await link.get_attribute("href")
                if not href or not _in_app_href(href, page_url):
                    continue
                key = href.split("?")[0].rstrip("/") or "/"
                if key in seen:
                    continue
                if not await link.is_visible():
                    continue
                box = await link.bounding_box()
                if not box or box["width"] < 4 or box["height"] < 4:
                    continue
                await link.scroll_into_view_if_needed()
                await link.click(timeout=12_000)
                seen.add(key)
                console.print(f"  [dim]Nav tour:[/] {href}")
                await asyncio.sleep(dwell)
                return True
            except Exception as e:
                console.print(f"[dim]nav tour skip: {e}[/]")
        return False

    async def click_role_nav_fallback() -> bool:
        """React Router <Link> may not use <nav>; match this app’s yellow nav labels."""
        for pat in (
            r"^VOTING$",
            r"^LEADERBOARD$",
            r"^UPLOAD$",
            r"whats\s*next",
        ):
            try:
                lk = page.get_by_role("link", name=re.compile(pat, re.I))
                if await lk.count() == 0:
                    continue
                el = lk.first
                if not await el.is_visible():
                    continue
                href = await el.get_attribute("href") or pat
                key = (href or "").split("?")[0].rstrip("/") or "/"
                if key in seen:
                    continue
                await el.scroll_into_view_if_needed()
                await el.click(timeout=12_000)
                seen.add(key)
                console.print(f"  [dim]Nav tour (role):[/] {pat.strip()} → {href}")
                await asyncio.sleep(dwell)
                return True
            except Exception:
                continue
        return False

    async def click_nav_dom_fallback() -> bool:
        """Last resort: click <a> whose text matches known labels (SPA router links)."""
        try:
            href = await page.evaluate("""() => {
              const labels = ['LEADERBOARD', 'UPLOAD', 'VOTING', 'WHATS NEXT'];
              const links = Array.from(document.querySelectorAll('a[href]'));
              for (const lab of labels) {
                const a = links.find((x) => {
                  const t = (x.innerText || '').replace(/\\s+/g, ' ').trim().toUpperCase();
                  return t === lab || t.includes(lab);
                });
                const h = a && a.getAttribute('href');
                if (a && h && !h.startsWith('#') && !h.startsWith('mailto:')) {
                  a.click();
                  return h;
                }
              }
              return '';
            }""")
            if href:
                key = href.split("?")[0].rstrip("/") or "/"
                if key in seen:
                    return False
                seen.add(key)
                console.print(f"  [dim]Nav tour (DOM):[/] {href}")
                await asyncio.sleep(dwell)
                return True
        except Exception:
            pass
        return False

    for round_i in range(14):
        ok = await click_href_links()
        if not ok:
            ok = await click_role_nav_fallback()
        if not ok:
            ok = await click_nav_dom_fallback()
        if not ok:
            if round_i == 0 and not seen:
                console.print(
                    "  [yellow]Nav tour:[/] no links found — "
                    "welcome modal may still be blocking, or nav uses unexpected markup."
                )
            break


async def _scroll_page_slowly(page: Page, pause: float = 1.2) -> None:
    """Smooth scroll down the full page height, pause, scroll back up."""
    scroll_height = await page.evaluate("document.body.scrollHeight")
    viewport_height = (page.viewport_size or {}).get("height", 720)
    if scroll_height <= viewport_height + 50:
        return  # Page doesn't scroll

    # Scroll down in steps for smooth visual
    steps = max(2, min(6, scroll_height // viewport_height))
    step_px = scroll_height // steps
    for i in range(1, steps + 1):
        await page.evaluate(f"window.scrollTo({{top: {step_px * i}, behavior: 'smooth'}})")
        await asyncio.sleep(0.6)
    await asyncio.sleep(pause)

    # Scroll back to top
    await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    await asyncio.sleep(0.8)


async def _zoom_into_section(page: Page, selector: str) -> None:
    """Zoom into a content section by scaling the viewport around it."""
    selectors = [s.strip() for s in selector.split(",")]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            box = await loc.bounding_box()
            if not box or box["width"] < 50 or box["height"] < 50:
                continue
            # Scroll element into view and apply a CSS zoom
            await loc.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)
            await page.evaluate("""(sel) => {
                const el = document.querySelector(sel);
                if (!el) return;
                el.style.transition = 'transform 0.6s ease';
                el.style.transformOrigin = 'center center';
                el.style.transform = 'scale(1.25)';
            }""", sel)
            await asyncio.sleep(2.0)
            # Reset zoom
            await page.evaluate("""(sel) => {
                const el = document.querySelector(sel);
                if (!el) return;
                el.style.transform = 'scale(1)';
            }""", sel)
            await asyncio.sleep(0.5)
            console.print(f"  [dim]Zoomed into:[/] {sel}")
            return
        except Exception:
            continue


async def _tour_navbar_deep(page: Page, dwell_ms: int) -> None:
    """Visit each nav route, scrolling through each page and clicking elements."""
    dwell = max(1.5, dwell_ms / 1000.0)
    seen: set[str] = set()
    await asyncio.sleep(1.0)

    # Collect all nav links first
    links = page.locator(
        "nav a[href], header a[href], [role='navigation'] a[href], "
        "a[href][class*='underline']"
    )
    try:
        n = await links.count()
    except Exception:
        n = 0

    page_url = page.url
    hrefs: list[str] = []
    for i in range(n):
        try:
            href = await links.nth(i).get_attribute("href")
            if href and _in_app_href(href, page_url):
                key = href.split("?")[0].rstrip("/") or "/"
                if key not in seen:
                    seen.add(key)
                    hrefs.append(href)
        except Exception:
            continue

    # Visit each route
    for href in hrefs:
        try:
            link = page.locator(f"a[href='{href}']").first
            if await link.count() == 0:
                continue
            if not await link.is_visible():
                continue
            await link.scroll_into_view_if_needed()
            await link.click(timeout=12_000)
            console.print(f"  [dim]Nav tour:[/] {href}")
            await asyncio.sleep(dwell)

            # Scroll through this page
            await _scroll_page_slowly(page, pause=0.8)

            # Click a couple of interactive elements on this page
            await _explore_clickables(page, max_actions=3)

        except Exception as e:
            console.print(f"  [dim]Nav tour skip: {e}[/]")

    if not hrefs:
        # Fallback to the original tour_navbar
        await _tour_navbar(page, dwell_ms)


async def _explore_clickables(page: Page, max_actions: int) -> None:
    """Click interactive elements. Includes div.cursor-pointer (common for React “cards”)."""
    if max_actions <= 0:
        return
    done = 0

    # Prefer #root so we hit app chrome, not extension junk; fall back to page.
    roots = [page.locator("#root"), page.locator("body")]
    click_selectors = [
        "button:not([disabled]):visible",
        "[role='button']:not([aria-disabled='true']):visible",
        "div.cursor-pointer:visible",
        "a[href^='/']:visible",
        "[data-testid]:visible",
        "input:not([type='hidden']):visible",
        "textarea:visible",
    ]
    seen_sig: set[str] = set()

    for root in roots:
        if done >= max_actions:
            break
        for sel in click_selectors:
            if done >= max_actions:
                break
            loc = root.locator(sel)
            try:
                n = await loc.count()
            except Exception:
                continue
            for i in range(min(n, 6)):
                if done >= max_actions:
                    break
                el = loc.nth(i)
                try:
                    if not await el.is_visible():
                        continue
                    href = await el.get_attribute("href")
                    if href in ("#", ""):
                        continue
                    tag = (await el.evaluate("node => node.tagName")).upper()
                    dedupe_key = f"{sel}|{i}|{tag}"
                    if dedupe_key in seen_sig:
                        continue
                    seen_sig.add(dedupe_key)

                    if tag in ("INPUT", "TEXTAREA"):
                        itype = (await el.get_attribute("type") or "text").lower()
                        if itype in ("submit", "button", "checkbox", "radio", "file", "image", "hidden"):
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
                        await asyncio.sleep(1.4)
                        continue

                    await el.scroll_into_view_if_needed()
                    await el.click(timeout=9_000)
                    done += 1
                    await asyncio.sleep(2.1)
                except Exception as e:
                    console.print(f"[dim]explore skip ({sel}): {e}[/]")


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
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(video_dir),
            record_video_size={"width": width, "height": height},
            # E2B (and similar) preview URLs must not stop on TLS interstitials.
            ignore_https_errors=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
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
        await page.goto(action.value, wait_until="domcontentloaded", timeout=120_000)
        await page.wait_for_load_state("load")
        await _wait_spa_ready(page)

    elif action.action == "dismiss_modals":
        await _dismiss_blocking_overlays(page)
        await asyncio.sleep(0.35)

    elif action.action == "tour_navbar":
        try:
            dwell = int((action.value or "3200").strip())
        except ValueError:
            dwell = 3200
        await _tour_navbar(page, dwell)

    elif action.action == "tour_navbar_deep":
        try:
            dwell = int((action.value or "3000").strip())
        except ValueError:
            dwell = 3000
        await _tour_navbar_deep(page, dwell)

    elif action.action == "zoom_section":
        await _zoom_into_section(page, action.value)

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
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
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
