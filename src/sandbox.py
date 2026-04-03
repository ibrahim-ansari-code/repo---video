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
    def __init__(
        self,
        manifest: RepoManifest,
        api_key: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.manifest = manifest
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "E2B API key required. Set E2B_API_KEY env var or pass --e2b-key.\n"
                "Get a free key at https://e2b.dev"
            )
        self._sandbox = None
        self._env = env or {}

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
        self._inject_env_vars()
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

    def _inject_env_vars(self) -> None:
        """Write user-provided env vars into the sandbox as a .env file and export them."""
        if not self._env:
            return
        console.print(f"  [dim]Injecting {len(self._env)} environment variable(s)[/]")
        # Write .env file for frameworks that read it (Vite, Next.js, etc.)
        env_lines = "\n".join(f"{k}={v}" for k, v in self._env.items())
        self._sandbox.files.write("/home/user/project/.env", env_lines + "\n")
        # Also export them in the shell profile so they're available to all commands
        export_lines = "\n".join(f"export {k}={v}" for k, v in self._env.items())
        self._sandbox.commands.run(
            f"echo '{export_lines}' >> /home/user/.bashrc",
            timeout=5,
        )

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

    def _patch_dev_server_config(self) -> None:
        """Patch Vite config to allow external hosts (E2B sandbox URLs).

        Vite 5.4+ blocks requests from non-localhost Host headers, returning 403.
        E2B exposes ports via {port}-{sandbox_id}.e2b.app URLs which Vite rejects.
        Fix: wrap the existing config to inject server.allowedHosts = true.
        """
        pt = self.manifest.project_type
        if pt not in (ProjectType.REACT_VITE, ProjectType.VUE):
            return

        # Find the existing vite config file
        result = self._sandbox.commands.run(
            "ls /home/user/project/vite.config.* 2>/dev/null | head -1",
            timeout=5,
        )
        config_file = (result.stdout or "").strip()
        if not config_file:
            # No vite config found, write a minimal one
            self._sandbox.files.write(
                "/home/user/project/vite.config.js",
                "import { defineConfig } from 'vite';\n"
                "export default defineConfig({ server: { allowedHosts: true } });\n",
            )
            return

        # Wrap the existing config to preserve plugins while adding allowedHosts.
        # Move original to _original.vite.config.{ext} so the import path is clean.
        ext = config_file.rsplit(".", 1)[-1]  # "ts" or "js"
        original_name = f"/home/user/project/_original.vite.config.{ext}"
        self._sandbox.commands.run(f"mv {config_file} {original_name}", timeout=5)
        self._sandbox.files.write(
            config_file,
            f"import originalConfig from './_original.vite.config';\n"
            "import { defineConfig, mergeConfig } from 'vite';\n"
            "export default mergeConfig(\n"
            "  typeof originalConfig === 'function' ? originalConfig({}) : originalConfig,\n"
            "  defineConfig({ server: { allowedHosts: true } }),\n"
            ");\n",
        )
        console.print("  [dim]Patched vite config with allowedHosts: true[/]")

    def _start_server(self, port: int) -> None:
        """Start the web server as a background process."""
        pt = self.manifest.project_type

        self._patch_dev_server_config()

        # Use manifest port for health checks / public URL; commands must listen on the same port.
        if pt == ProjectType.REACT_VITE:
            start_cmd = f"npx vite --host 0.0.0.0 --port {port}"
        elif pt == ProjectType.NEXTJS:
            start_cmd = f"npx next dev -p {port}"
        elif pt == ProjectType.VUE:
            start_cmd = f"npx vue-cli-service serve --port {port}"
        else:
            start_cmd = START_COMMANDS.get(pt)

        if start_cmd is None:
            start_cmd = self.manifest.run_command or "npm start"

        console.print(f"  [dim]Starting server: {start_cmd}[/]")
        # Long-lived dev servers must run in the background; a foreground run hits E2B cmd timeout.
        try:
            self._sandbox.commands.run(
                f"cd /home/user/project && {start_cmd}",
                background=True,
                timeout=60,
            )
        except Exception as exc:
            # E2B v2.19 raises CommandExitException on non-zero exit from background
            # processes. The server may have crashed on startup.
            console.print(f"  [yellow]Warning:[/] Background server start raised: {exc}")

        self._wait_for_server(port)

    def _wait_for_server(self, port: int) -> None:
        """Poll until the server inside the sandbox is responding."""
        console.print(f"  [dim]Waiting for port {port}...[/]")
        deadline = time.time() + BOOT_WAIT_TIMEOUT

        while time.time() < deadline:
            # E2B raises on non-zero exit; curl fails until the server is up — probe must exit 0.
            result = self._sandbox.commands.run(
                f"bash -c 'curl -sf --connect-timeout 2 http://127.0.0.1:{port}/ -o /dev/null && echo OK || true'",
                timeout=15,
            )
            if (result.stdout or "").strip() == "OK":
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
