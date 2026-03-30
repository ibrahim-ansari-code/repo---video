"""Tests for anecdote prompt generation and parsing."""

from __future__ import annotations

import json

import pytest

from src.anecdote.prompts import (
    build_anecdote_prompt,
    parse_anecdote_response,
    FALLBACK_KEYFRAMES,
)


class TestBuildPrompt:
    def test_includes_project_info(self):
        prompt = build_anecdote_prompt("myapp", "A cool tool", "node", "# MyApp\nDoes stuff")
        assert "myapp" in prompt
        assert "A cool tool" in prompt
        assert "node" in prompt

    def test_truncates_long_readme(self):
        long_readme = "x" * 5000
        prompt = build_anecdote_prompt("app", "desc", "python", long_readme)
        assert len(prompt) < 5000


class TestParseResponse:
    def test_valid_json(self):
        response = json.dumps({
            "scenario": "A developer is frustrated",
            "keyframes": ["prompt1", "prompt2", "prompt3"],
            "motion_prompts": ["motion1", "motion2", "motion3"],
            "overlay_text": "Ever had this problem?",
        })
        result = parse_anecdote_response(response)
        assert len(result["keyframes"]) == 3
        assert result["overlay_text"] == "Ever had this problem?"

    def test_json_in_markdown(self):
        response = "Here is the result:\n```json\n" + json.dumps({
            "scenario": "test",
            "keyframes": ["a", "b"],
            "motion_prompts": ["c", "d"],
        }) + "\n```"
        result = parse_anecdote_response(response)
        assert len(result["keyframes"]) == 2

    def test_fallback_on_invalid(self):
        result = parse_anecdote_response("not json at all")
        assert result["keyframes"] == list(FALLBACK_KEYFRAMES)
