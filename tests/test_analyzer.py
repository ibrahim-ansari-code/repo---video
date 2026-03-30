"""Tests for the repo analyzer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.analyzer import (
    ProjectType,
    RepoManifest,
    detect_project_type,
    parse_readme,
    _extract_code_blocks,
    _extract_demo_hints,
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal fake repo directory."""
    return tmp_path


class TestDetectProjectType:
    def test_nextjs(self, tmp_repo):
        (tmp_repo / "package.json").write_text(json.dumps({"dependencies": {"next": "14.0.0"}}))
        (tmp_repo / "next.config.js").write_text("module.exports = {}")
        pt, meta = detect_project_type(tmp_repo)
        assert pt == ProjectType.NEXTJS
        assert meta["port"] == 3000

    def test_react_vite(self, tmp_repo):
        (tmp_repo / "package.json").write_text(json.dumps({"dependencies": {"react": "18.0.0"}}))
        (tmp_repo / "vite.config.ts").write_text("export default {}")
        pt, meta = detect_project_type(tmp_repo)
        assert pt == ProjectType.REACT_VITE
        assert meta["port"] == 5173

    def test_python_flask(self, tmp_repo):
        (tmp_repo / "requirements.txt").write_text("flask==3.0.0")
        app = tmp_repo / "app.py"
        app.write_text("from flask import Flask\napp = Flask(__name__)")
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.PYTHON_FLASK

    def test_python_fastapi(self, tmp_repo):
        (tmp_repo / "requirements.txt").write_text("fastapi==0.100.0")
        app = tmp_repo / "main.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.PYTHON_FASTAPI

    def test_python_django(self, tmp_repo):
        (tmp_repo / "manage.py").write_text("#!/usr/bin/env python")
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.PYTHON_DJANGO

    def test_rust(self, tmp_repo):
        (tmp_repo / "Cargo.toml").write_text('[package]\nname = "myapp"')
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.RUST

    def test_go(self, tmp_repo):
        (tmp_repo / "go.mod").write_text("module myapp")
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.GO

    def test_docker(self, tmp_repo):
        (tmp_repo / "Dockerfile").write_text("FROM ubuntu:22.04")
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.DOCKER

    def test_unknown(self, tmp_repo):
        pt, _ = detect_project_type(tmp_repo)
        assert pt == ProjectType.UNKNOWN


class TestParseReadme:
    def test_basic_readme(self, tmp_repo):
        (tmp_repo / "README.md").write_text(
            "# My Cool Project\n\n"
            "A tool that does amazing things for developers.\n\n"
            "## Getting Started\n\n"
            "```bash\nnpm install\nnpm start\n```\n\n"
            "## Usage\n\n"
            "```bash\nmyapp --help\nmyapp run\n```\n"
        )
        desc, examples, hints = parse_readme(tmp_repo)
        assert "amazing things" in desc
        assert len(examples) >= 2
        assert any("usage" in h.lower() for h in hints)

    def test_no_readme(self, tmp_repo):
        desc, examples, hints = parse_readme(tmp_repo)
        assert desc == ""
        assert examples == []
        assert hints == []


class TestExtractCodeBlocks:
    def test_bash_blocks(self):
        content = "Some text\n```bash\nnpm install\nnpm start\n```\nMore text\n```shell\npython app.py\n```"
        blocks = _extract_code_blocks(content)
        assert len(blocks) == 2
        assert "npm install" in blocks[0]
        assert "python app.py" in blocks[1]

    def test_no_blocks(self):
        assert _extract_code_blocks("No code here") == []


class TestExtractDemoHints:
    def test_finds_headings(self):
        content = "# Title\n## Quick Start\n## Features\n## API Reference\n## Usage Examples\n"
        hints = _extract_demo_hints(content)
        assert len(hints) >= 2
        assert any("Quick Start" in h for h in hints)
        assert any("Usage" in h for h in hints)
