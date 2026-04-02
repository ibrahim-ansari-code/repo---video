"""Stage 1: Repo Analyzer — clone, detect project type, parse README, extract description."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from git import Repo
from rich.console import Console

console = Console()


class ProjectType(str, Enum):
    NEXTJS = "nextjs"
    REACT_VITE = "react_vite"
    VUE = "vue"
    NODE = "node"
    PYTHON_FLASK = "python_flask"
    PYTHON_DJANGO = "python_django"
    PYTHON_FASTAPI = "python_fastapi"
    PYTHON_GENERIC = "python_generic"
    RUST = "rust"
    GO = "go"
    DOCKER = "docker"
    UNKNOWN = "unknown"


WEB_APP_TYPES = {
    ProjectType.NEXTJS,
    ProjectType.REACT_VITE,
    ProjectType.VUE,
    ProjectType.NODE,
    ProjectType.PYTHON_FLASK,
    ProjectType.PYTHON_DJANGO,
    ProjectType.PYTHON_FASTAPI,
}


@dataclass
class RepoManifest:
    repo_url: str
    clone_dir: Path
    project_type: ProjectType
    name: str
    description: str
    setup_commands: list[str] = field(default_factory=list)
    run_command: str = ""
    port: int = 3000
    is_web_app: bool = False
    readme_content: str = ""
    usage_examples: list[str] = field(default_factory=list)
    demo_hints: list[str] = field(default_factory=list)
    # package.json "description" — safe for title cards; avoids README install steps.
    package_description: str = ""


def clone_repo(repo_url: str, target_dir: Path | None = None) -> Path:
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="repovideo_"))
    console.print(f"[bold blue]Cloning[/] {repo_url} → {target_dir}")
    Repo.clone_from(repo_url, str(target_dir), depth=1)
    return target_dir


def detect_project_type(repo_dir: Path) -> tuple[ProjectType, dict]:
    """Scan marker files to determine the project type and gather metadata."""
    markers: dict[str, bool] = {}
    check_files = [
        "package.json", "next.config.js", "next.config.mjs", "next.config.ts",
        "vite.config.js", "vite.config.ts", "vue.config.js", "nuxt.config.ts",
        "requirements.txt", "pyproject.toml", "setup.py", "manage.py",
        "Pipfile", "Cargo.toml", "go.mod", "Dockerfile", "docker-compose.yml",
    ]
    for f in check_files:
        markers[f] = (repo_dir / f).exists()

    pkg_json = {}
    if markers["package.json"]:
        import json
        try:
            pkg_json = json.loads((repo_dir / "package.json").read_text())
        except Exception:
            pass

    deps = {**pkg_json.get("dependencies", {}), **pkg_json.get("devDependencies", {})}

    if any(markers.get(f) for f in ["next.config.js", "next.config.mjs", "next.config.ts"]):
        return ProjectType.NEXTJS, {"port": 3000, "pkg": pkg_json}
    if markers.get("nuxt.config.ts") or "vue" in deps:
        return ProjectType.VUE, {"port": 3000, "pkg": pkg_json}
    if any(markers.get(f) for f in ["vite.config.js", "vite.config.ts"]):
        return ProjectType.REACT_VITE, {"port": 5173, "pkg": pkg_json}
    if markers["manage.py"]:
        return ProjectType.PYTHON_DJANGO, {"port": 8000}
    if markers["requirements.txt"] or markers["pyproject.toml"] or markers["setup.py"]:
        pyfiles = _scan_python_imports(repo_dir)
        if "fastapi" in pyfiles:
            return ProjectType.PYTHON_FASTAPI, {"port": 8000}
        if "flask" in pyfiles:
            return ProjectType.PYTHON_FLASK, {"port": 5000}
        return ProjectType.PYTHON_GENERIC, {}
    if markers["Cargo.toml"]:
        return ProjectType.RUST, {}
    if markers["go.mod"]:
        return ProjectType.GO, {}
    if markers["package.json"]:
        return ProjectType.NODE, {"port": 3000, "pkg": pkg_json}
    if markers["Dockerfile"] or markers["docker-compose.yml"]:
        return ProjectType.DOCKER, {}
    return ProjectType.UNKNOWN, {}


def _scan_python_imports(repo_dir: Path) -> set[str]:
    """Quick scan of .py files for known framework imports."""
    imports: set[str] = set()
    for py_file in repo_dir.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")[:5000]
        except Exception:
            continue
        for framework in ("flask", "fastapi", "django", "streamlit"):
            if re.search(rf"(?:from|import)\s+{framework}", content):
                imports.add(framework)
    return imports


def _readme_line_poor_description(line: str) -> bool:
    """Skip setup / install lines when picking a README summary for the manifest."""
    s = line.strip()
    if len(s) < 12:
        return True
    low = s.lower()
    if re.match(r"^\d+\.\s", s):
        return True
    if any(
        p in low
        for p in (
            "install depend",
            "npm install",
            "yarn install",
            "pnpm install",
            "clone the",
            "git clone",
            "getting started",
            "```",
            "development server",
            "build the",
        )
    ):
        return True
    return False


def parse_readme(repo_dir: Path) -> tuple[str, list[str], list[str]]:
    """Extract description, usage examples, and demo hints from README."""
    readme_path = None
    for name in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
        candidate = repo_dir / name
        if candidate.exists():
            readme_path = candidate
            break

    if readme_path is None:
        return "", [], []

    content = readme_path.read_text(errors="ignore")

    lines = content.strip().split("\n")
    description = ""
    found_title = False
    for line in lines:
        raw = line.strip()
        is_heading = raw.startswith("#")
        stripped = raw.lstrip("#").strip()
        if not stripped or len(stripped) <= 10:
            continue
        if is_heading and not found_title:
            found_title = True
            continue
        if _readme_line_poor_description(stripped):
            continue
        description = stripped
        break

    usage_examples = _extract_code_blocks(content)
    demo_hints = _extract_demo_hints(content)

    return description, usage_examples, demo_hints


def _extract_code_blocks(content: str) -> list[str]:
    """Pull fenced code blocks that look like shell commands."""
    blocks: list[str] = []
    pattern = re.compile(r"```(?:bash|sh|shell|console|zsh)?\s*\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(content):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks[:10]


def _extract_demo_hints(content: str) -> list[str]:
    """Look for section headers that suggest demo-worthy features."""
    hints: list[str] = []
    pattern = re.compile(r"^#+\s+(.+)", re.MULTILINE)
    for match in pattern.finditer(content):
        heading = match.group(1).strip().lower()
        keywords = ("usage", "example", "demo", "getting started", "quick start", "features")
        if any(kw in heading for kw in keywords):
            hints.append(match.group(1).strip())
    return hints


def _build_setup_commands(project_type: ProjectType, meta: dict) -> list[str]:
    if project_type in (ProjectType.NEXTJS, ProjectType.REACT_VITE, ProjectType.VUE, ProjectType.NODE):
        return ["npm install"]
    if project_type in (ProjectType.PYTHON_FLASK, ProjectType.PYTHON_DJANGO,
                        ProjectType.PYTHON_FASTAPI, ProjectType.PYTHON_GENERIC):
        return ["pip install -r requirements.txt"]
    if project_type == ProjectType.RUST:
        return ["cargo build --release"]
    if project_type == ProjectType.GO:
        return ["go build -o app ."]
    return []


def _build_run_command(project_type: ProjectType, meta: dict) -> str:
    pkg = meta.get("pkg", {})
    scripts = pkg.get("scripts", {})

    if project_type == ProjectType.NEXTJS:
        return scripts.get("dev", "npm run dev")
    if project_type == ProjectType.REACT_VITE:
        return scripts.get("dev", "npm run dev")
    if project_type == ProjectType.VUE:
        return scripts.get("dev", scripts.get("serve", "npm run dev"))
    if project_type == ProjectType.NODE:
        return scripts.get("start", "npm start")
    if project_type == ProjectType.PYTHON_DJANGO:
        return "python manage.py runserver 0.0.0.0:8000"
    if project_type == ProjectType.PYTHON_FLASK:
        return "flask run --host=0.0.0.0 --port=5000"
    if project_type == ProjectType.PYTHON_FASTAPI:
        return "uvicorn main:app --host 0.0.0.0 --port 8000"
    if project_type == ProjectType.RUST:
        return "./target/release/app"
    if project_type == ProjectType.GO:
        return "./app"
    return ""


def _get_port(project_type: ProjectType, meta: dict) -> int:
    if "port" in meta:
        return meta["port"]
    return 3000


def analyze_repo(repo_url: str, clone_dir: Path | None = None) -> RepoManifest:
    """Full analysis pipeline: clone → detect → parse → build manifest."""
    repo_dir = clone_repo(repo_url, clone_dir)
    project_type, meta = detect_project_type(repo_dir)
    readme_desc, usage_examples, demo_hints = parse_readme(repo_dir)
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    pkg = meta.get("pkg") or {}
    pd = pkg.get("description")
    package_description = pd.strip() if isinstance(pd, str) else ""
    if len(package_description) > 8:
        description = package_description
    else:
        description = readme_desc

    console.print(f"[bold green]Detected[/] {project_type.value} project: [bold]{name}[/]")

    return RepoManifest(
        repo_url=repo_url,
        clone_dir=repo_dir,
        project_type=project_type,
        name=name,
        description=description,
        package_description=package_description,
        setup_commands=_build_setup_commands(project_type, meta),
        run_command=_build_run_command(project_type, meta),
        port=_get_port(project_type, meta),
        is_web_app=project_type in WEB_APP_TYPES,
        readme_content=(repo_dir / "README.md").read_text(errors="ignore")
        if (repo_dir / "README.md").exists()
        else "",
        usage_examples=usage_examples,
        demo_hints=demo_hints,
    )
