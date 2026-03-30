"""Stage 3: Docker Sandbox Runner — build, run, and manage containerized repos."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import BuildError, ContainerError, ImageNotFound
from rich.console import Console

from src.analyzer import ProjectType, RepoManifest, WEB_APP_TYPES
from src.config import DOCKERFILES_DIR

console = Console()

RESOURCE_LIMITS = {
    "mem_limit": "2g",
    "cpu_period": 100_000,
    "cpu_quota": 100_000,  # 1 CPU
}

TIMEOUT_SECONDS = 120


@dataclass
class SandboxResult:
    container_id: str
    host_port: int | None
    is_web: bool
    run_command: str | None  # for CLI tools, the command to exec


class Sandbox:
    def __init__(self, manifest: RepoManifest):
        self.manifest = manifest
        self.client = docker.from_env()
        self.container = None
        self.image_tag = f"repovideo-{manifest.name}:latest"

    def _generate_dockerfile(self) -> str:
        """Build a Dockerfile string for the detected project type."""
        pt = self.manifest.project_type

        if pt == ProjectType.DOCKER:
            existing = self.manifest.clone_dir / "Dockerfile"
            if existing.exists():
                return existing.read_text()

        template_map = {
            ProjectType.NEXTJS: "node.Dockerfile",
            ProjectType.REACT_VITE: "node.Dockerfile",
            ProjectType.VUE: "node.Dockerfile",
            ProjectType.NODE: "node.Dockerfile",
            ProjectType.PYTHON_FLASK: "python.Dockerfile",
            ProjectType.PYTHON_DJANGO: "python.Dockerfile",
            ProjectType.PYTHON_FASTAPI: "python.Dockerfile",
            ProjectType.PYTHON_GENERIC: "python.Dockerfile",
            ProjectType.RUST: "rust.Dockerfile",
            ProjectType.GO: "go.Dockerfile",
        }

        template_name = template_map.get(pt)
        if template_name is None:
            return self._fallback_dockerfile()

        template_path = DOCKERFILES_DIR / template_name
        if not template_path.exists():
            return self._fallback_dockerfile()

        content = template_path.read_text()
        content = self._customize_dockerfile(content, pt)
        return content

    def _customize_dockerfile(self, content: str, pt: ProjectType) -> str:
        """Adjust the template Dockerfile for the specific project."""
        port = self.manifest.port

        if pt == ProjectType.REACT_VITE:
            content = content.replace("EXPOSE 3000", f"EXPOSE {port}")
            content = content.replace('CMD ["npm", "start"]', f'CMD ["npx", "vite", "--host", "0.0.0.0", "--port", "{port}"]')
        elif pt == ProjectType.NEXTJS:
            content = content.replace('CMD ["npm", "start"]', 'CMD ["npx", "next", "dev", "-p", "3000"]')
        elif pt == ProjectType.VUE:
            content = content.replace('CMD ["npm", "start"]', 'CMD ["npx", "vue-cli-service", "serve", "--port", "3000"]')
        elif pt == ProjectType.PYTHON_FASTAPI:
            content = content.replace(
                'CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]',
                'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]',
            )
        elif pt == ProjectType.PYTHON_DJANGO:
            content = content.replace(
                'CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]',
                'CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]',
            )

        return content

    def _fallback_dockerfile(self) -> str:
        return (
            "FROM ubuntu:22.04\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            'CMD ["bash"]\n'
        )

    def build(self) -> None:
        """Write the Dockerfile into the clone dir and build the image."""
        dockerfile_content = self._generate_dockerfile()
        dockerfile_path = self.manifest.clone_dir / "Dockerfile.repovideo"
        dockerfile_path.write_text(dockerfile_content)

        console.print(f"[bold blue]Building[/] Docker image [bold]{self.image_tag}[/]")
        try:
            self.client.images.build(
                path=str(self.manifest.clone_dir),
                dockerfile="Dockerfile.repovideo",
                tag=self.image_tag,
                rm=True,
            )
        except BuildError as e:
            console.print(f"[bold red]Build failed:[/] {e}")
            raise

    def start(self) -> SandboxResult:
        """Run the container, returning connection details."""
        is_web = self.manifest.project_type in WEB_APP_TYPES
        port = self.manifest.port

        run_kwargs = {
            "image": self.image_tag,
            "detach": True,
            "remove": False,
            **RESOURCE_LIMITS,
        }

        if is_web:
            run_kwargs["ports"] = {f"{port}/tcp": None}

        console.print(f"[bold blue]Starting[/] container for [bold]{self.manifest.name}[/]")
        self.container = self.client.containers.run(**run_kwargs)

        host_port = None
        if is_web:
            host_port = self._wait_for_port(port)
            console.print(f"[bold green]Ready[/] at http://localhost:{host_port}")

        return SandboxResult(
            container_id=self.container.id,
            host_port=host_port,
            is_web=is_web,
            run_command=self.manifest.run_command if not is_web else None,
        )

    def _wait_for_port(self, container_port: int, timeout: int = TIMEOUT_SECONDS) -> int:
        """Poll until the container's port mapping is available and the server responds."""
        import urllib.request
        import urllib.error

        deadline = time.time() + timeout
        host_port = None

        while time.time() < deadline:
            self.container.reload()
            ports = self.container.ports
            mapping = ports.get(f"{container_port}/tcp")
            if mapping:
                host_port = int(mapping[0]["HostPort"])
                break
            time.sleep(1)

        if host_port is None:
            raise TimeoutError(f"Container port {container_port} not mapped after {timeout}s")

        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{host_port}", timeout=2)
                return host_port
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)

        console.print("[yellow]Warning:[/] Server may not be fully ready, proceeding anyway")
        return host_port

    def exec_command(self, command: str) -> tuple[int, str]:
        """Execute a command inside the running container."""
        if self.container is None:
            raise RuntimeError("Container not started")
        exit_code, output = self.container.exec_run(command, demux=True)
        stdout = output[0].decode() if output[0] else ""
        stderr = output[1].decode() if output[1] else ""
        return exit_code, stdout + stderr

    def get_logs(self) -> str:
        if self.container is None:
            return ""
        return self.container.logs().decode(errors="ignore")

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def cleanup(self) -> None:
        """Full cleanup: stop container + remove clone dir."""
        self.stop()
        try:
            self.client.images.remove(self.image_tag, force=True)
        except Exception:
            pass
