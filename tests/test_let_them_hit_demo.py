"""Integration tests against the real let-them-hit sample repo.

`.env` is loaded from the repo root in `conftest.py` (optional `python-dotenv`).
- Without network: skip integration clone tests.
- Without `E2B_API_KEY`: E2B construction / smoke tests are skipped.
- Full `Sandbox.start()` runs only when `E2B_API_KEY` and `REPVIDEO_E2B_SMOKE=1` are set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.analyzer import ProjectType, analyze_repo
from src.recorder.script_generator import generate_web_demo_script

LET_THEM_HIT_REPO = "https://github.com/ibrahim-ansari-code/let-them-hit.git"


requires_network = pytest.mark.skipif(
    os.environ.get("REPVIDEO_SKIP_NETWORK") == "1",
    reason="REPVIDEO_SKIP_NETWORK=1",
)


@requires_network
@pytest.mark.integration
def test_let_them_hit_analyze_clone_and_detect(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    manifest = analyze_repo(LET_THEM_HIT_REPO, clone_dir=clone_root)

    assert manifest.repo_url == LET_THEM_HIT_REPO
    assert manifest.project_type == ProjectType.REACT_VITE
    assert manifest.is_web_app is True
    assert manifest.name == "let-them-hit"
    assert manifest.clone_dir == clone_root
    assert manifest.port == 5173


@requires_network
@pytest.mark.integration
def test_let_them_hit_manifest_setup_and_run_commands(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    manifest = analyze_repo(LET_THEM_HIT_REPO, clone_dir=clone_root)

    assert manifest.setup_commands == ["npm install"]
    assert manifest.run_command
    rc = manifest.run_command.lower()
    assert "dev" in rc or "vite" in rc


@requires_network
@pytest.mark.integration
def test_let_them_hit_web_demo_script_from_manifest(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    manifest = analyze_repo(LET_THEM_HIT_REPO, clone_dir=clone_root)

    app_url = f"http://127.0.0.1:{manifest.port}"
    script = generate_web_demo_script(manifest, app_url=app_url)

    assert script.url == app_url
    assert len(script.actions) >= 3
    navigates = [a for a in script.actions if a.action == "navigate"]
    assert navigates, "expected at least one navigate action"
    assert navigates[0].value == app_url


@pytest.mark.e2b
def test_let_them_hit_sandbox_requires_key_or_env(tmp_path: Path) -> None:
    """Sandbox reads API key from env when not passed explicitly (after .env load)."""
    pytest.importorskip("e2b")
    from src.sandbox import Sandbox

    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    (clone_root / "package.json").write_text('{"name":"x","scripts":{"dev":"vite"}}')
    (clone_root / "vite.config.ts").write_text("export default {}\n")

    from src.analyzer import RepoManifest

    manifest = RepoManifest(
        repo_url=LET_THEM_HIT_REPO,
        clone_dir=clone_root,
        project_type=ProjectType.REACT_VITE,
        name="let-them-hit",
        description="test",
        port=5173,
        is_web_app=True,
    )

    key = os.environ.get("E2B_API_KEY", "").strip()
    if not key:
        with pytest.raises(RuntimeError, match="E2B API key"):
            Sandbox(manifest)
        return

    box = Sandbox(manifest)
    assert box.api_key == key


@requires_network
@pytest.mark.integration
@pytest.mark.e2b
@pytest.mark.slow
def test_let_them_hit_e2b_full_pipeline_smoke(tmp_path: Path) -> None:
    """Creates E2B VM, clones repo, npm install, starts Vite (no video). Opt-in: REPVIDEO_E2B_SMOKE=1."""
    pytest.importorskip("e2b")

    if os.environ.get("REPVIDEO_E2B_SMOKE", "").strip() != "1":
        pytest.skip("Set REPVIDEO_E2B_SMOKE=1 in .env to run this test (slow; uses E2B quota)")

    key = os.environ.get("E2B_API_KEY", "").strip()
    if not key:
        pytest.skip("E2B_API_KEY not set")

    from src.sandbox import Sandbox

    clone_root = tmp_path / "analyze"
    clone_root.mkdir()
    manifest = analyze_repo(LET_THEM_HIT_REPO, clone_dir=clone_root)

    box = Sandbox(manifest, api_key=key)
    try:
        result = box.start()
        assert result.is_web is True
        assert result.host_url and result.host_url.startswith("https://")
        assert result.sandbox_id
    finally:
        box.cleanup()
