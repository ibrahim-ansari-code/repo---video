"""Tests for the video compositor utilities."""

from __future__ import annotations

import pytest

from src.compositor import _escape_ffmpeg_text


class TestEscapeText:
    def test_escapes_colons(self):
        assert "\\:" in _escape_ffmpeg_text("http://example.com")

    def test_escapes_single_quotes(self):
        assert "\\'" in _escape_ffmpeg_text("it's a test")

    def test_escapes_percent(self):
        assert "%%" in _escape_ffmpeg_text("100%")

    def test_escapes_backslash(self):
        assert "\\\\" in _escape_ffmpeg_text("path\\to\\file")

    def test_plain_text_unchanged(self):
        assert _escape_ffmpeg_text("hello world") == "hello world"
