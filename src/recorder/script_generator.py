"""Auto-generate demo scripts from README content and project type."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.analyzer import ProjectType, RepoManifest


@dataclass
class DemoAction:
    action: str  # "navigate", "click", "type", "scroll", "wait", "screenshot"
    selector: str = ""
    value: str = ""
    wait_ms: int = 1000
    description: str = ""


@dataclass
class DemoScript:
    actions: list[DemoAction] = field(default_factory=list)
    url: str = ""


def generate_web_demo_script(manifest: RepoManifest, app_url: str | None = None, host_port: int | None = None) -> DemoScript:
    """Build a sequence of browser actions to demo a web app."""
    base_url = app_url or f"http://localhost:{host_port}"
    script = DemoScript(url=base_url)

    script.actions.append(DemoAction(
        action="navigate",
        value=base_url,
        wait_ms=3000,
        description="Load the homepage",
    ))

    script.actions.append(DemoAction(
        action="wait",
        wait_ms=2000,
        description="Let the page fully render",
    ))

    script.actions.append(DemoAction(
        action="screenshot",
        description="Capture the initial state",
    ))

    routes = _extract_routes_from_readme(manifest.readme_content)
    for route in routes[:5]:
        url = f"{base_url}{route}" if route.startswith("/") else f"{base_url}/{route}"
        script.actions.append(DemoAction(
            action="navigate",
            value=url,
            wait_ms=2000,
            description=f"Navigate to {route}",
        ))
        script.actions.append(DemoAction(
            action="screenshot",
            description=f"Capture {route}",
        ))

    script.actions += _generate_interaction_actions(manifest)

    script.actions.append(DemoAction(
        action="scroll",
        value="bottom",
        wait_ms=1500,
        description="Scroll to the bottom of the page",
    ))

    script.actions.append(DemoAction(
        action="scroll",
        value="top",
        wait_ms=1500,
        description="Scroll back to top",
    ))

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
    """Find URL paths mentioned in README."""
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


def _generate_interaction_actions(manifest: RepoManifest) -> list[DemoAction]:
    """Generate generic interaction actions based on project type."""
    actions: list[DemoAction] = []

    actions.append(DemoAction(
        action="click",
        selector="a[href]:not([href^='http']):not([href^='#'])",
        wait_ms=2000,
        description="Click the first internal navigation link",
    ))

    actions.append(DemoAction(
        action="click",
        selector="button:visible:first-of-type",
        wait_ms=1500,
        description="Click the first visible button",
    ))

    form_actions = [
        DemoAction(
            action="type",
            selector="input[type='text']:first-of-type, input[type='email']:first-of-type",
            value="demo@example.com",
            wait_ms=500,
            description="Type into the first text input",
        ),
        DemoAction(
            action="click",
            selector="button[type='submit'], input[type='submit']",
            wait_ms=2000,
            description="Submit the form",
        ),
    ]
    actions.extend(form_actions)

    return actions


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
    pt = manifest.project_type
    if pt == ProjectType.RUST:
        return ["./target/release/app --help", "./target/release/app"]
    if pt == ProjectType.GO:
        return ["./app --help", "./app"]
    if pt == ProjectType.PYTHON_GENERIC:
        return ["python -m app --help", "python -m app"]
    return ["echo 'Demo: running the project'"]
