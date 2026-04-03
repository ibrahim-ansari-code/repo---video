"""Auto-generate demo scripts — in-browser exploration, not README replay."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.analyzer import ProjectType, RepoManifest


@dataclass
class DemoAction:
    action: str  # navigate, dismiss_modals, tour_navbar, explore_ui, ...
    selector: str = ""
    value: str = ""
    wait_ms: int = 1000
    description: str = ""


@dataclass
class DemoScript:
    actions: list[DemoAction] = field(default_factory=list)
    url: str = ""


def generate_web_demo_script(
    manifest: RepoManifest,
    app_url: str | None = None,
    host_port: int | None = None,
) -> DemoScript:
    """Drive the live app: load, wait for UI, click buttons/links, use inputs, scroll.

    Does not navigate to paths guessed from the README — that often 404s or shows
    the wrong screen for SPAs. Exploration uses real DOM controls on the current page.
    """
    base_url = app_url or f"http://localhost:{host_port}"
    script = DemoScript(url=base_url)

    script.actions.append(
        DemoAction(
            action="navigate",
            value=base_url,
            wait_ms=600,
            description="Open the running app",
        )
    )

    # Full-screen modals (e.g. “START”, cookie banners) block nav — dismiss first.
    script.actions.append(
        DemoAction(
            action="dismiss_modals",
            wait_ms=1200,
            description="Close welcome / overlay dialogs",
        )
    )

    # React Router / SPA: walk real nav links so each route gets screen time.
    # Tour each nav route. After each route, the recorder will auto-scroll
    # and explore the page (see tour_navbar_deep action).
    script.actions.append(
        DemoAction(
            action="tour_navbar_deep",
            value="3000",
            wait_ms=0,
            description="Visit each route: scroll through page, click elements, then next",
        )
    )

    # Final exploration pass on the landing page
    script.actions.append(
        DemoAction(
            action="navigate",
            value=base_url,
            wait_ms=800,
            description="Return to home for final exploration",
        )
    )

    script.actions.append(
        DemoAction(
            action="explore_ui",
            value="10",
            wait_ms=0,
            description="Click main-content controls (cards, buttons, inputs)",
        )
    )

    script.actions.append(
        DemoAction(
            action="zoom_section",
            value="main, #root > div > div, .container, [class*='content']",
            wait_ms=2500,
            description="Zoom in on main content area",
        )
    )

    script.actions.append(
        DemoAction(
            action="scroll",
            value="bottom",
            wait_ms=1800,
            description="Scroll down to show more content",
        )
    )
    script.actions.append(
        DemoAction(
            action="scroll",
            value="top",
            wait_ms=1200,
            description="Scroll back to top",
        )
    )

    return script


def generate_cli_demo_script(manifest: RepoManifest) -> list[str]:
    """Extract CLI commands from README to replay in the terminal."""
    commands: list[str] = []

    for block in manifest.usage_examples:
        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("$"):
                line = line[1:].strip()
            if line.startswith("#") or not line:
                continue
            if _is_safe_command(line):
                commands.append(line)

    if not commands:
        commands = _generate_default_cli_commands(manifest)

    return commands[:15]


def _extract_routes_from_readme(content: str) -> list[str]:
    """Find URL paths mentioned in README (used by tests / optional tooling)."""
    routes: list[str] = []
    pattern = re.compile(r"(?:localhost[:\d]*|127\.0\.0\.1[:\d]*)(\/[a-zA-Z0-9/_-]+)")
    for match in pattern.finditer(content):
        route = match.group(1)
        if route not in routes:
            routes.append(route)

    path_pattern = re.compile(r"`(\/[a-zA-Z0-9/_-]{2,})`")
    for match in path_pattern.finditer(content):
        route = match.group(1)
        if route not in routes and not route.startswith("/usr") and not route.startswith("/etc"):
            routes.append(route)

    return routes


def _is_safe_command(cmd: str) -> bool:
    """Filter out dangerous or install-only commands."""
    dangerous = ["rm ", "sudo", "chmod", "chown", "mkfs", "dd ", "curl | sh", "wget | sh"]
    install_only = ["npm install", "pip install", "yarn install", "cargo build", "go build"]
    for d in dangerous:
        if d in cmd:
            return False
    for i in install_only:
        if cmd.strip().startswith(i):
            return False
    return True


def _generate_default_cli_commands(manifest: RepoManifest) -> list[str]:
    """Fallback commands when README doesn't have usage examples."""
    name = manifest.name or "app"
    pt = manifest.project_type
    if pt == ProjectType.RUST:
        return [f"./target/release/{name} --help", f"./target/release/{name}"]
    if pt == ProjectType.GO:
        return [f"./{name} --help", f"./{name}"]
    if pt == ProjectType.PYTHON_GENERIC:
        return [f"python -m {name} --help", f"python -m {name}"]
    return ["echo 'Demo: running the project'"]
