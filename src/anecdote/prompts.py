"""Prompt templates for generating anecdote scenarios from repo descriptions."""

from __future__ import annotations

ANECDOTE_SYSTEM_PROMPT = """You are a creative director for developer tool demo videos.
Given a project description, generate a relatable real-world scenario that shows WHY
someone would need this tool. The scenario should be:
- Visual and concrete (describable in a single image)
- Emotionally relatable (frustration, wasted time, confusion)
- Brief — one scene that sets up the problem

Output a JSON object with these fields:
- "scenario": A 1-2 sentence description of the problem scenario
- "keyframes": A list of 3-5 image generation prompts, each describing a single frame.
  Each prompt should be detailed, cinematic, and suitable for Stable Diffusion XL.
  Include lighting, composition, mood, and style directions.
- "motion_prompts": A list matching keyframes, describing the motion/animation for each
  (e.g., "camera slowly pans right", "person shakes head in frustration")
- "overlay_text": Short text to overlay on the anecdote segment (e.g., "Ever had this problem?")
"""

ANECDOTE_USER_PROMPT = """Project: {name}
Description: {description}
Type: {project_type}
README excerpt: {readme_excerpt}

Generate a compelling anecdote scenario for this project's demo video intro."""


FALLBACK_KEYFRAMES = [
    (
        "A developer sitting at a cluttered desk late at night, multiple browser tabs open on "
        "dual monitors, coffee cup empty, warm desk lamp lighting, photorealistic, cinematic "
        "shallow depth of field, frustration visible on their face, 4K quality"
    ),
    (
        "Close-up of a laptop screen showing a wall of error messages and red warning icons, "
        "reflection of a tired developer visible in the screen, dramatic lighting from the "
        "monitor, cyberpunk aesthetic, ultra detailed"
    ),
    (
        "A developer throwing their hands up in frustration, papers and sticky notes scattered "
        "on a desk, dramatic side lighting, motion blur on hands, editorial photography style, "
        "8K resolution"
    ),
]

FALLBACK_MOTION_PROMPTS = [
    "Camera slowly zooms in on the developer's face, subtle head shake",
    "Screen content slowly scrolling down, flickering light from monitor",
    "Hands move upward in frustration, slight camera shake",
]

FALLBACK_OVERLAY = "Sound familiar?"


def build_anecdote_prompt(name: str, description: str, project_type: str, readme: str) -> str:
    excerpt = readme[:1500] if readme else "No README available"
    return ANECDOTE_USER_PROMPT.format(
        name=name,
        description=description,
        project_type=project_type,
        readme_excerpt=excerpt,
    )


def parse_anecdote_response(response: str) -> dict:
    """Parse the LLM response into structured anecdote data."""
    import json
    import re

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "keyframes" in data and "motion_prompts" in data:
                return data
        except json.JSONDecodeError:
            pass

    return {
        "scenario": "A developer struggling with a common problem",
        "keyframes": list(FALLBACK_KEYFRAMES),
        "motion_prompts": list(FALLBACK_MOTION_PROMPTS),
        "overlay_text": FALLBACK_OVERLAY,
    }
