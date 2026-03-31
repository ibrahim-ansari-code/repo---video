"""Stage 3: E2B Cloud Sandbox — clone, install, and run repos in isolated VMs.

Uses E2B (https://e2b.dev) cloud sandboxes instead of local Docker containers.
Each sandbox is a lightweight Firecracker microVM that boots in ~150ms with
full Linux, network access, and port forwarding.

Requires an E2B API key: set E2B_API_KEY env var or pass it explicitly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from src.analyzer import ProjectType, RepoManifest, WEB_APP_TYPES

console = Console()

SANDBOX_TIMEOUT = 300  # 5 minutes
BOOT_WAIT_TIMEOUT = 120  # max seconds to wait for server to respond


@dataclass
class SandboxResult:
    sandbox_id: str
    host_url: str | None
    is_web: bool
    run_command: str | None


INSTALL_COMMANDS: dict[ProjectType, list[str]] = {
    ProjectType.NEXTJS: ["npm install"],
    ProjectType.REACT_VITE: ["npm install"],
    ProjectType.VUE: ["npm install"],
    ProjectType.NODE: ["npm install"],
    ProjectType.PYTHON_FLASK: ["pip install -r requirements.txt 2>/dev/null || pip install flask"],
    ProjectType.PYTHON_DJANGO: ["pip install -r requirements.txt 2>/dev/null || pip install django"],
    ProjectType.PYTHON_FASTAPI: ["pip install -r requirements.txt 2>/dev/null || pip install fastapi uvicorn"],
    ProjectType.PYTHON_GENERIC: ["pip install -r requirements.txt 2>/dev/null || true"],
    ProjectType.RUST: ["cargo build --release 2>/dev/null || true"],
    ProjectType.GO: ["go build ./... 2>/dev/null || true"],
}

START_COMMANDS: dict[ProjectType, str] = {
    ProjectType.NEXTJS: "npx next dev -p 3000",
    ProjectType.REACT_VITE: "npx vite --host 0.0.0.0 --port 3000",
    ProjectType.VUE: "npx vue-cli-service serve --port 3000",
    ProjectType.NODE: "npm start",
    ProjectType.PYTHON_FLASK: "python -m flask run --host=0.0.0.0 --port=8000",
    ProjectType.PYTHON_DJANGO: "python manage.py runserver 0.0.0.0:8000",
    ProjectType.PYTHON_FASTAPI: "uvicorn main:app --host 0.0.0.0 --port 8000",
}


class Sandbox:
    def __init__(self, manifest: RepoManifest, api_key: str | None = None):
        self.manifest = manifest
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "E2B API key required. Set E2B_API_KEY env var or pass --e2b-key.\n"
                "Get a free key at https://e2b.dev"
            )
        self._sandbox = None

    def start(self) -> SandboxResult:
        """Create an E2B sandbox, clone the repo, install deps, and start the app."""
        from e2b import Sandbox as E2BSandbox

        is_web = self.manifest.project_type in WEB_APP_TYPES
        port = self.manifest.port

        console.print(f"[bold blue]Creating[/] E2B sandbox for [bold]{self.manifest.name}[/]")

        self._sandbox = E2BSandbox.create(
            api_key=self.api_key,
            timeout=SANDBOX_TIMEOUT,
        )

        console.print(f"  [dim]Sandbox ID: {self._sandbox.sandbox_id}[/]")

        self._clone_repo()
        self._install_deps()

        host_url = None
        if is_web:
            self._start_server(port)
            host_url = self._get_public_url(port)
            console.print(f"[bold green]Ready[/] at {host_url}")

        return SandboxResult(
            sandbox_id=self._sandbox.sandbox_id,
            host_url=host_url,
            is_web=is_web,
            run_command=self.manifest.run_command if not is_web else None,
        )

    def _clone_repo(self) -> None:
        """Clone the repo inside the sandbox."""
        console.print(f"  [dim]Cloning {self.manifest.repo_url}...[/]")
        result = self._sandbox.commands.run(
            f"git clone --depth 1 {self.manifest.repo_url} /home/user/project",
            timeout=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Clone failed: {result.stderr}")

    def _install_deps(self) -> None:
        """Install project dependencies based on detected type."""
        pt = self.manifest.project_type
        commands = INSTALL_COMMANDS.get(pt, [])

        if not commands:
            return

        console.print(f"  [dim]Installing dependencies ({pt.value})...[/]")
        for cmd in commands:
            result = self._sandbox.commands.run(
                cmd,
                timeout=120,
                cwd="/home/user/project",
            )
            if result.exit_code != 0:
                console.print(f"  [yellow]Warning:[/] '{cmd}' exited with {result.exit_code}")

    def _start_server(self, port: int) -> None:
        """Start the web server as a background process."""
        pt = self.manifest.project_type
        start_cmd = START_COMMANDS.get(pt)

        if start_cmd is None:
            start_cmd = self.manifest.run_command or "npm start"

        console.print(f"  [dim]Starting server: {start_cmd}[/]")
        self._sandbox.commands.run(
            f"cd /home/user/project && {start_cmd} &",
            timeout=5,
        )

        self._wait_for_server(port)

    def _wait_for_server(self, port: int) -> None:
        """Poll until the server inside the sandbox is responding."""
        console.print(f"  [dim]Waiting for port {port}...[/]")
        deadline = time.time() + BOOT_WAIT_TIMEOUT

        while time.time() < deadline:
            result = self._sandbox.commands.run(
                f"curl -sf http://localhost:{port}/ -o /dev/null -w '%{{http_code}}'",
                timeout=5,
            )
            if result.exit_code == 0:
                return
            time.sleep(2)

        console.print("[yellow]Warning:[/] Server may not be fully ready, proceeding anyway")

    def _get_public_url(self, port: int) -> str:
        """Get the public URL for a port exposed from the sandbox."""
        host = self._sandbox.get_host(port)
        return f"https://{host}"

    def exec_command(self, command: str) -> tuple[int, str]:
        """Execute a command inside the running sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        result = self._sandbox.commands.run(
            command,
            timeout=30,
            cwd="/home/user/project",
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.exit_code, output

    def get_logs(self) -> str:
        """Get recent output from the sandbox."""
        if self._sandbox is None:
            return ""
        result = self._sandbox.commands.run("cat /tmp/server.log 2>/dev/null || true", timeout=5)
        return result.stdout or ""

    def stop(self) -> None:
        """Kill the E2B sandbox."""
        if self._sandbox:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None

    def cleanup(self) -> None:
        """Full cleanup — just stop the sandbox (E2B handles the rest)."""
        self.stop()
