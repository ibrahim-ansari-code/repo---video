"""Unit tests for YouTube-Commons dataset helpers (no network)."""

from __future__ import annotations

from src.anecdote.datasets import (
    _caption_from_youtube_commons_row,
    _yt_dlp_section,
    _youtube_commons_watch_url,
    get_dataset_tier,
)


class TestYouTubeCommonsWatchUrl:
    def test_prefers_video_link(self):
        u = _youtube_commons_watch_url({
            "video_link": "https://www.youtube.com/watch?v=abc123",
            "video_id": "ignored",
        })
        assert u == "https://www.youtube.com/watch?v=abc123"

    def test_short_link(self):
        u = _youtube_commons_watch_url({"video_link": "https://youtu.be/xyz789", "video_id": ""})
        assert "youtu.be" in u

    def test_fallback_video_id(self):
        u = _youtube_commons_watch_url({"video_link": "", "video_id": "dQw4w9WgXcQ"})
        assert u == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_missing(self):
        assert _youtube_commons_watch_url({"video_link": "", "video_id": ""}) is None


class TestYtDlpSection:
    def test_twelve_seconds(self):
        assert _yt_dlp_section(12) == "*0:00-0:12"

    def test_ninety_seconds(self):
        assert _yt_dlp_section(90) == "*0:00-1:30"


class TestCaptionFromRow:
    def test_title_and_text(self):
        c = _caption_from_youtube_commons_row({"title": "Hello", "text": "World " * 100}, max_chars=50)
        assert "Hello" in c
        assert c.endswith("...")


def test_youtube_commons_tier():
    assert get_dataset_tier("youtube-commons") == "video"
