# repovideo

Take any GitHub repo, run it in a sandbox, and generate a polished demo video — complete with an AI-generated intro anecdote.

## Quick Start

```bash
pip install -e .
playwright install chromium

# Generate a demo video from any repo
repovideo https://github.com/user/repo

# Fine-tune the video style with LoRA first
repovideo https://github.com/user/repo --train-lora ./reference-videos/

# Skip the AI anecdote intro
repovideo https://github.com/user/repo --no-anecdote

# Custom output path and demo duration
repovideo https://github.com/user/repo -o my-demo.mp4 --duration 60
```

## Requirements

- Python 3.11+
- Docker (for sandboxed repo execution)
- FFmpeg (for video compositing)
- GPU with 8-24GB VRAM (for AI video generation; falls back to CPU)

## Pipeline

1. **Analyze** — Clone the repo, detect project type, parse README
2. **Anecdote** — Generate an AI intro video showing why the project matters
3. **Sandbox** — Run the project safely in Docker
4. **Record** — Capture a demo (browser for web apps, terminal for CLIs)
5. **Composite** — Stitch everything into a single polished MP4

## Architecture

```
RepoURL → Analyzer → Sandbox Runner → Demo Recorder ─┐
                  └→ Anecdote Generator ──────────────┤
                                                      └→ Compositor → Final MP4
```
