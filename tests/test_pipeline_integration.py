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
        assert any(a.action == "explore_ui" for a in script.actions)
        assert any(a.action == "dismiss_modals" for a in script.actions)
        assert any(a.action == "tour_navbar_deep" for a in script.actions)
        nav_values = [a.value for a in script.actions if a.action == "navigate"]
        assert "http://localhost:3000" in nav_values

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


class TestWanPipelineDtype:
    """Verify that the shared Wan pipeline loader sets correct dtypes per component."""

    @patch("diffusers.WanImageToVideoPipeline.from_pretrained")
    @patch("diffusers.AutoencoderKLWan.from_pretrained")
    @patch("transformers.CLIPVisionModel.from_pretrained")
    def test_vae_and_encoder_use_float32(self, mock_clip, mock_vae, mock_pipe):
        """VAE and image encoder must be float32; pipeline dtype is passed through."""
        import torch
        from src.anecdote.video_gen import load_wan_i2v_pipeline

        mock_clip.return_value = MagicMock()
        mock_vae.return_value = MagicMock()
        mock_pipe.return_value = MagicMock()

        load_wan_i2v_pipeline("test-model", dtype=torch.bfloat16, cache_dir="/tmp")

        # VAE loaded in float32
        _, vae_kwargs = mock_vae.call_args
        assert vae_kwargs["torch_dtype"] == torch.float32

        # Image encoder loaded in float32
        _, clip_kwargs = mock_clip.call_args
        assert clip_kwargs["torch_dtype"] == torch.float32

        # Pipeline uses the requested dtype (bfloat16 for transformer)
        _, pipe_kwargs = mock_pipe.call_args
        assert pipe_kwargs["torch_dtype"] == torch.bfloat16
        assert pipe_kwargs["vae"] is mock_vae.return_value
        assert pipe_kwargs["image_encoder"] is mock_clip.return_value

    @patch("src.anecdote.video_gen.load_wan_i2v_pipeline")
    @patch("src.anecdote.video_gen._get_device", return_value="cpu")
    @patch("src.anecdote.video_gen.load_wan_peft_lora_state_dict")
    def test_load_wan_pipeline_with_lora(self, mock_lora_load, mock_device, mock_loader, tmp_path):
        """LoRA weights are loaded when a valid path is provided."""
        from src.anecdote.video_gen import _load_wan_pipeline

        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe  # .to(device) returns same object
        mock_loader.return_value = mock_pipe
        mock_lora_load.return_value = {"key": "value"}

        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "adapter_model.safetensors").write_bytes(b"fake")

        _load_wan_pipeline("14B", lora_dir)

        mock_lora_load.assert_called_once()
        mock_pipe.load_lora_weights.assert_called_once()


class TestSandboxConfigPatching:
    """Verify Vite config patching for E2B sandbox host allowlisting."""

    def test_patch_vite_config_for_react_vite(self, fake_web_repo):
        from src.sandbox import Sandbox

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=fake_web_repo,
            project_type=ProjectType.REACT_VITE,
            name="test",
            description="test",
            port=3000,
            is_web_app=True,
        )
        sandbox = Sandbox(manifest, api_key="e2b_test_key_123")

        mock_sb = MagicMock()
        mock_sb.commands.run.return_value = MagicMock(
            stdout="/home/user/project/vite.config.ts", exit_code=0
        )
        sandbox._sandbox = mock_sb

        sandbox._patch_dev_server_config()

        # Should have moved original and written new config
        mock_sb.commands.run.assert_any_call(
            "mv /home/user/project/vite.config.ts /home/user/project/_original.vite.config.ts",
            timeout=5,
        )
        mock_sb.files.write.assert_called_once()
        written_content = mock_sb.files.write.call_args[0][1]
        assert "allowedHosts: true" in written_content
        assert "_original.vite.config" in written_content

    def test_patch_skips_nextjs(self, fake_web_repo):
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

        mock_sb = MagicMock()
        sandbox._sandbox = mock_sb

        sandbox._patch_dev_server_config()

        # Should not write any files for Next.js
        mock_sb.files.write.assert_not_called()

    def test_background_cmd_exception_does_not_crash(self, fake_web_repo):
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

        mock_sb = MagicMock()

        def run_side_effect(cmd, timeout=60, background=None, **_kwargs):
            if background:
                raise RuntimeError("CommandExitException: exit code -1")
            # curl health probe in _wait_for_server
            if "curl" in cmd and "127.0.0.1" in cmd:
                return MagicMock(stdout="OK\n", exit_code=0)
            return MagicMock(stdout="", exit_code=0)

        mock_sb.commands.run.side_effect = run_side_effect
        sandbox._sandbox = mock_sb

        # Should not raise — background start exception is caught and logged
        sandbox._start_server(3000)


class TestCliDefaultCommands:
    def test_default_commands_use_manifest_name(self):
        from src.recorder.script_generator import _generate_default_cli_commands

        manifest = RepoManifest(
            repo_url="https://github.com/user/test",
            clone_dir=Path("/tmp/test"),
            project_type=ProjectType.RUST,
            name="ripgrep",
            description="test",
            port=0,
            is_web_app=False,
        )
        commands = _generate_default_cli_commands(manifest)
        assert any("ripgrep" in cmd for cmd in commands)
        assert not any("/app" in cmd for cmd in commands)


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
