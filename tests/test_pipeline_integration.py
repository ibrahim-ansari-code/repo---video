"""Integration tests for the full pipeline.

These tests mock external dependencies (Docker, GPU models, ffmpeg)
to verify the pipeline wiring works end-to-end.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import ProjectType, RepoManifest, analyze_repo, detect_project_type
from src.config import PipelineConfig


@pytest.fixture
def fake_web_repo(tmp_path):
    """Create a fake Next.js repo on disk."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "test-web-app",
        "scripts": {"dev": "next dev"},
        "dependencies": {"next": "14.0.0", "react": "18.0.0"},
    }))
    (tmp_path / "next.config.js").write_text("module.exports = {}")
    (tmp_path / "README.md").write_text(
        "# Test Web App\n\n"
        "A sample web application for testing repovideo.\n\n"
        "## Getting Started\n\n"
        "```bash\nnpm install\nnpm run dev\n```\n\n"
        "Visit http://localhost:3000/dashboard\n"
    )
    return tmp_path


@pytest.fixture
def fake_cli_repo(tmp_path):
    """Create a fake Rust CLI repo on disk."""
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "test-cli"\nversion = "0.1.0"'
    )
    (tmp_path / "README.md").write_text(
        "# Test CLI\n\n"
        "A sample CLI tool for testing repovideo.\n\n"
        "## Usage\n\n"
        "```bash\n$ test-cli --help\n$ test-cli process input.csv\n```\n"
    )
    return tmp_path


class TestAnalyzerIntegration:
    def test_web_repo_detection(self, fake_web_repo):
        pt, meta = detect_project_type(fake_web_repo)
        assert pt == ProjectType.NEXTJS
        assert meta["port"] == 3000

    def test_cli_repo_detection(self, fake_cli_repo):
        pt, _ = detect_project_type(fake_cli_repo)
        assert pt == ProjectType.RUST


class TestSandboxIntegration:
    def test_sandbox_requires_api_key(self, fake_web_repo):
        from src.sandbox import Sandbox

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=fake_web_repo,
            project_type=ProjectType.NEXTJS,
            name="test",
            description="test",
            port=3000,
            is_web_app=True,
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("E2B_API_KEY", None)
            with pytest.raises(RuntimeError, match="E2B API key"):
                Sandbox(manifest)

    def test_sandbox_accepts_api_key(self, fake_web_repo):
        from src.sandbox import Sandbox

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=fake_web_repo,
            project_type=ProjectType.NEXTJS,
            name="test",
            description="test",
            port=3000,
            is_web_app=True,
        )
        sandbox = Sandbox(manifest, api_key="e2b_test_key_123")
        assert sandbox.api_key == "e2b_test_key_123"

    def test_install_commands_for_project_types(self):
        from src.sandbox import INSTALL_COMMANDS, START_COMMANDS

        assert ProjectType.NEXTJS in INSTALL_COMMANDS
        assert "npm install" in INSTALL_COMMANDS[ProjectType.NEXTJS]
        assert ProjectType.NEXTJS in START_COMMANDS
        assert "next" in START_COMMANDS[ProjectType.NEXTJS]
        assert ProjectType.PYTHON_FASTAPI in START_COMMANDS
        assert "uvicorn" in START_COMMANDS[ProjectType.PYTHON_FASTAPI]


class TestScriptGeneratorIntegration:
    def test_web_script_from_manifest(self, fake_web_repo):
        from src.recorder.script_generator import generate_web_demo_script

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=fake_web_repo,
            project_type=ProjectType.NEXTJS,
            name="test-web-app",
            description="A sample web app",
            port=3000,
            is_web_app=True,
            readme_content=(fake_web_repo / "README.md").read_text(),
            usage_examples=["npm install\nnpm run dev"],
            demo_hints=["Getting Started"],
        )
        script = generate_web_demo_script(manifest, app_url="http://localhost:3000")
        assert len(script.actions) > 0
        nav_values = [a.value for a in script.actions if a.action == "navigate"]
        assert any("dashboard" in v for v in nav_values)

    def test_cli_script_from_manifest(self, fake_cli_repo):
        from src.recorder.script_generator import generate_cli_demo_script

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=fake_cli_repo,
            project_type=ProjectType.RUST,
            name="test-cli",
            description="A sample CLI tool",
            port=0,
            is_web_app=False,
            readme_content=(fake_cli_repo / "README.md").read_text(),
            usage_examples=["$ test-cli --help\n$ test-cli process input.csv"],
            demo_hints=["Usage"],
        )
        commands = generate_cli_demo_script(manifest)
        assert len(commands) > 0
        assert any("test-cli" in cmd for cmd in commands)


class TestConfigValidation:
    def test_default_config(self):
        config = PipelineConfig(repo_url="https://github.com/user/repo")
        assert config.output_path == Path("output.mp4")
        assert config.demo_duration == 60
        assert config.no_anecdote is False
        assert config.width == 1920
        assert config.height == 1080

    def test_custom_config(self):
        config = PipelineConfig(
            repo_url="https://github.com/user/repo",
            output_path=Path("custom.mp4"),
            demo_duration=30,
            no_anecdote=True,
            model_size="480P",
        )
        assert config.output_path == Path("custom.mp4")
        assert config.demo_duration == 30
        assert config.no_anecdote is True
        assert config.model_size == "480P"
