"""Tests for the demo script generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analyzer import ProjectType, RepoManifest
from src.recorder.script_generator import (
    generate_web_demo_script,
    generate_cli_demo_script,
    _extract_routes_from_readme,
    _is_safe_command,
)


@pytest.fixture
def web_manifest():
    return RepoManifest(
        repo_url="https://github.com/user/web-app",
        clone_dir=Path("/tmp/fake"),
        project_type=ProjectType.NEXTJS,
        name="web-app",
        description="A sample web application",
        setup_commands=["npm install"],
        run_command="npm run dev",
        port=3000,
        is_web_app=True,
        readme_content="## Usage\n\nVisit http://localhost:3000/dashboard\n",
        usage_examples=["npm run dev"],
        demo_hints=["Usage"],
    )


@pytest.fixture
def cli_manifest():
    return RepoManifest(
        repo_url="https://github.com/user/cli-tool",
        clone_dir=Path("/tmp/fake"),
        project_type=ProjectType.RUST,
        name="cli-tool",
        description="A command-line tool",
        setup_commands=["cargo build --release"],
        run_command="./target/release/app",
        port=0,
        is_web_app=False,
        readme_content="## Usage\n\n```bash\n$ mytool --help\n$ mytool process data.csv\n```\n",
        usage_examples=["$ mytool --help\n$ mytool process data.csv"],
        demo_hints=["Usage"],
    )


class TestWebDemoScript:
    def test_generates_actions(self, web_manifest):
        script = generate_web_demo_script(web_manifest, app_url="http://localhost:3000")
        assert len(script.actions) > 0
        assert script.url == "http://localhost:3000"
        action_types = [a.action for a in script.actions]
        assert "navigate" in action_types

    def test_includes_ui_exploration_not_readme_routes(self, web_manifest):
        script = generate_web_demo_script(web_manifest, app_url="http://localhost:3000")
        types = [a.action for a in script.actions]
        assert "explore_ui" in types
        assert "dismiss_modals" in types
        assert "tour_navbar_deep" in types
        nav = [a.value for a in script.actions if a.action == "navigate"]
        assert "http://localhost:3000" in nav


class TestCliDemoScript:
    def test_extracts_commands(self, cli_manifest):
        commands = generate_cli_demo_script(cli_manifest)
        assert len(commands) > 0
        assert any("mytool" in cmd for cmd in commands)

    def test_strips_dollar_prefix(self, cli_manifest):
        commands = generate_cli_demo_script(cli_manifest)
        assert not any(cmd.startswith("$") for cmd in commands)


class TestExtractRoutes:
    def test_finds_localhost_routes(self):
        content = "Visit http://localhost:3000/api/users and http://localhost:3000/dashboard"
        routes = _extract_routes_from_readme(content)
        assert "/api/users" in routes
        assert "/dashboard" in routes

    def test_finds_backtick_routes(self):
        content = "The endpoint is `/api/v1/items`"
        routes = _extract_routes_from_readme(content)
        assert "/api/v1/items" in routes


class TestSafetyFilter:
    def test_blocks_dangerous(self):
        assert not _is_safe_command("rm -rf /")
        assert not _is_safe_command("sudo apt install foo")

    def test_blocks_install_only(self):
        assert not _is_safe_command("npm install")
        assert not _is_safe_command("pip install flask")

    def test_allows_safe(self):
        assert _is_safe_command("mytool --help")
        assert _is_safe_command("echo hello")
        assert _is_safe_command("python app.py")
