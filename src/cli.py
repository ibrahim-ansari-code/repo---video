"""CLI entry point — wire all pipeline stages together."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.config import PipelineConfig, ensure_dirs

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """repovideo — generate polished demo videos from any repo."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("repo_url")
@click.option("-o", "--output", default="output.mp4", help="Output video path")
@click.option("--duration", default=60, help="Max demo recording duration in seconds")
@click.option("--no-anecdote", is_flag=True, help="Skip AI anecdote generation")
@click.option("--train-lora", type=click.Path(exists=True), help="Dir of reference videos to fine-tune LoRA")
@click.option("--lora-name", default=None, help="Name for the LoRA adapter (or existing one to load)")
@click.option("--model-size", type=click.Choice(["14B", "480P"]), default="14B", help="Wan2.1 I2V resolution (both use 14B weights)")
@click.option("--width", default=1920, help="Output video width")
@click.option("--height", default=1080, help="Output video height")
@click.option("--colab", is_flag=True, help="Offload GPU work to Google Colab via Google Drive")
@click.option("--gdrive-path", type=click.Path(), default=None, help="Path to Google Drive sync folder")
@click.option("--anecdote-file", type=click.Path(exists=True), default=None, help="Use a pre-made anecdote video (e.g. from Colab)")
@click.option(
    "--dataset",
    type=click.Choice(
        ["tip-i2v", "pusa", "cinematic", "pexels", "youtube-commons", "developer", "dramatic"]
    ),
    default=None,
    help="Built-in HF dataset to train LoRA on (instead of --train-lora)",
)
@click.option("--dataset-max-samples", type=int, default=None,
              help="Max samples to download from built-in dataset (e.g. 8 for a quick test)")
@click.option("--remote", default=None, help="URL of a remote GPU inference server (NodeOps, RunPod, etc.)")
@click.option(
    "--e2b-key",
    default=None,
    envvar="E2B_API_KEY",
    help="E2B API key for cloud sandbox (Stage 3). Free tier: https://e2b.dev — or set E2B_API_KEY.",
)
def generate(
    repo_url: str,
    output: str,
    duration: int,
    no_anecdote: bool,
    train_lora: str | None,
    lora_name: str | None,
    model_size: str,
    width: int,
    height: int,
    colab: bool,
    gdrive_path: str | None,
    anecdote_file: str | None,
    dataset: str | None,
    dataset_max_samples: int | None,
    remote: str | None,
    e2b_key: str | None,
):
    """Generate a demo video from a GitHub repo URL."""
    ensure_dirs()

    config = PipelineConfig(
        repo_url=repo_url,
        output_path=Path(output),
        demo_duration=duration,
        no_anecdote=no_anecdote,
        train_lora=Path(train_lora) if train_lora else None,
        lora_name=lora_name,
        model_size=model_size,
        width=width,
        height=height,
        use_colab=colab,
        gdrive_path=Path(gdrive_path) if gdrive_path else None,
        anecdote_file=Path(anecdote_file) if anecdote_file else None,
        dataset_name=dataset,
        dataset_max_samples=dataset_max_samples,
        remote_url=remote,
        e2b_api_key=e2b_key,
    )

    console.print(Panel(
        f"[bold]repovideo[/] — generating demo for [cyan]{repo_url}[/]",
        border_style="blue",
    ))

    try:
        _run_pipeline(config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        raise


@main.command()
def list_datasets():
    """List built-in Hugging Face datasets available for LoRA training."""
    from src.anecdote.datasets import list_builtin_datasets

    datasets = list_builtin_datasets()
    console.print("[bold]Built-in datasets for LoRA training:[/]\n")
    for ds in datasets:
        tier_map = {"i2v": ("bold green", "I2V"), "video": ("green", "VIDEO"), "image": ("yellow", "IMAGE-ONLY")}
        tier_color, tier_label = tier_map.get(ds["tier"], ("yellow", ds["tier"]))
        console.print(
            f"  [bold cyan]{ds['name']}[/] [{tier_color}][{tier_label}][/{tier_color}]"
        )
        console.print(f"    {ds['description']}")
        console.print(f"    {ds['samples']} default samples from [dim]{ds['hf_dataset']}[/]")
        console.print()
    console.print("[bold]Recommended:[/] Use [bold green]pusa[/] (VIDEO tier) for best LoRA results — real video clips with motion.")
    console.print("[dim]Usage: repovideo generate <url> --dataset pusa[/]")


@main.command()
def list_loras():
    """List all available trained LoRA adapters."""
    from src.anecdote.lora_trainer import list_available_loras

    ensure_dirs()
    loras = list_available_loras()
    if not loras:
        console.print("[dim]No LoRA adapters found. Train one with --train-lora[/]")
        return
    for lora in loras:
        console.print(f"  [bold]{lora.get('lora_name', 'unknown')}[/] — {lora.get('path', '')}")


def _run_pipeline(config: PipelineConfig) -> None:
    """Execute the full pipeline: analyze → anecdote → sandbox → record → composite."""
    from src.analyzer import analyze_repo
    from src.sandbox import Sandbox

    work_dir = Path(tempfile.mkdtemp(prefix="repovideo_pipeline_"))

    # --- Stage 1: Analyze ---
    console.rule("[bold]Stage 1: Analyzing Repo")
    manifest = analyze_repo(config.repo_url)
    console.print(f"  Type: [bold]{manifest.project_type.value}[/]")
    console.print(f"  Web app: [bold]{'Yes' if manifest.is_web_app else 'No'}[/]")
    console.print(f"  Description: {manifest.description[:100]}")

    # --- Stage 2: AI Anecdote (if enabled) ---
    anecdote_path = None
    overlay_text = "Sound familiar?"

    if config.anecdote_file:
        console.rule("[bold]Stage 2: Using Pre-made Anecdote")
        anecdote_path = config.anecdote_file
        console.print(f"  Using pre-made anecdote: [cyan]{anecdote_path}[/]")
    elif not config.no_anecdote:
        console.rule("[bold]Stage 2: Generating AI Anecdote")
        if config.remote_url:
            anecdote_path, overlay_text = _generate_anecdote_via_remote(config, manifest, work_dir)
        elif config.use_colab:
            anecdote_path, overlay_text = _generate_anecdote_via_colab(config, manifest, work_dir)
        else:
            anecdote_path, overlay_text = _generate_anecdote(config, manifest, work_dir)

    # --- Stage 2.5: LoRA training (if requested) ---
    lora_path = None
    if config.train_lora or config.dataset_name:
        console.rule("[bold]Stage 2.5: Training LoRA")
        if config.use_colab:
            lora_path = _train_lora_via_colab(config)
        else:
            lora_path = _train_lora(config)

    # --- Stage 3: Sandbox (E2B cloud) ---
    console.rule("[bold]Stage 3: Running in E2B Sandbox")
    sandbox = Sandbox(manifest, api_key=config.e2b_api_key)
    try:
        sandbox_result = sandbox.start()

        # --- Stage 4: Record ---
        console.rule("[bold]Stage 4: Recording Demo")
        demo_path = work_dir / "demo.mp4"

        if sandbox_result.is_web:
            from src.recorder.browser_recorder import run_web_recording
            run_web_recording(manifest, sandbox_result.host_url, demo_path, config.demo_duration)
        else:
            from src.recorder.terminal_recorder import record_cli_demo
            record_cli_demo(manifest, sandbox.exec_command, demo_path, config.demo_duration)

    finally:
        sandbox.stop()

    # --- Stage 5: Composite ---
    console.rule("[bold]Stage 5: Compositing Final Video")
    from src.compositor import composite_video, subtitle_for_title_card

    composite_video(
        demo_path=demo_path,
        output_path=config.output_path,
        project_name=manifest.name,
        project_description=subtitle_for_title_card(manifest.package_description),
        repo_url=config.repo_url,
        anecdote_path=anecdote_path,
        overlay_text=overlay_text,
        width=config.width,
        height=config.height,
        fps=config.fps,
    )

    console.print(Panel(
        f"[bold green]Done![/] Video saved to [cyan]{config.output_path}[/]",
        border_style="green",
    ))


def _generate_anecdote(config: PipelineConfig, manifest, work_dir: Path) -> tuple[Path | None, str]:
    """Generate the AI anecdote intro video."""
    from src.anecdote.prompts import (
        ANECDOTE_SYSTEM_PROMPT,
        FALLBACK_KEYFRAMES,
        FALLBACK_MOTION_PROMPTS,
        FALLBACK_OVERLAY,
        build_anecdote_prompt,
        parse_anecdote_response,
    )

    keyframes = list(FALLBACK_KEYFRAMES)
    motion_prompts = list(FALLBACK_MOTION_PROMPTS)
    overlay_text = FALLBACK_OVERLAY

    try:
        import litellm

        user_prompt = build_anecdote_prompt(
            manifest.name,
            manifest.description,
            manifest.project_type.value,
            manifest.readme_content,
        )

        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANECDOTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=1000,
        )

        parsed = parse_anecdote_response(response.choices[0].message.content)
        keyframes = parsed["keyframes"]
        motion_prompts = parsed["motion_prompts"]
        overlay_text = parsed.get("overlay_text", FALLBACK_OVERLAY)
        console.print(f"  [green]Generated anecdote scenario[/]")

    except Exception as e:
        console.print(f"  [yellow]LLM unavailable ({e}), using fallback prompts[/]")

    try:
        from src.anecdote.image_gen import generate_keyframes
        from src.anecdote.video_gen import generate_video_from_images

        image_paths = generate_keyframes(keyframes, work_dir / "keyframes")

        lora_path = None
        if config.lora_name:
            from src.config import LORA_DIR
            candidate = LORA_DIR / config.lora_name
            if candidate.exists():
                lora_path = candidate

        anecdote_path = work_dir / "anecdote.mp4"
        generate_video_from_images(
            image_paths,
            motion_prompts,
            anecdote_path,
            lora_path=lora_path,
            model_size=config.model_size,
        )

        return anecdote_path, overlay_text

    except Exception as e:
        console.print(f"  [yellow]AI video generation failed ({e}), skipping anecdote[/]")
        return None, overlay_text


def _train_lora(config: PipelineConfig) -> Path | None:
    """Run LoRA fine-tuning if requested."""
    from src.anecdote.lora_trainer import LoRATrainingConfig, train_lora

    try:
        lora_name = config.lora_name or config.dataset_name or "custom_style"
        training_config = LoRATrainingConfig(
            reference_dir=config.train_lora,
            dataset_name=config.dataset_name,
            dataset_max_samples=config.dataset_max_samples,
            lora_name=lora_name,
            model_size=config.model_size,
        )
        lora_dir = train_lora(training_config)
        return lora_dir
    except Exception as e:
        console.print(f"  [yellow]LoRA training failed ({e}), continuing without[/]")
        return None


def _generate_anecdote_via_colab(config: PipelineConfig, manifest, work_dir: Path) -> tuple[Path | None, str]:
    """Generate the AI anecdote by dispatching GPU work to Colab."""
    from src.anecdote.prompts import (
        ANECDOTE_SYSTEM_PROMPT,
        FALLBACK_KEYFRAMES,
        FALLBACK_MOTION_PROMPTS,
        FALLBACK_OVERLAY,
        build_anecdote_prompt,
        parse_anecdote_response,
    )
    from src.remote import submit_anecdote_job, download_anecdote_result

    keyframes = list(FALLBACK_KEYFRAMES)
    motion_prompts = list(FALLBACK_MOTION_PROMPTS)
    overlay_text = FALLBACK_OVERLAY

    try:
        import litellm

        user_prompt = build_anecdote_prompt(
            manifest.name,
            manifest.description,
            manifest.project_type.value,
            manifest.readme_content,
        )
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANECDOTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=1000,
        )
        parsed = parse_anecdote_response(response.choices[0].message.content)
        keyframes = parsed["keyframes"]
        motion_prompts = parsed["motion_prompts"]
        overlay_text = parsed.get("overlay_text", FALLBACK_OVERLAY)
        console.print(f"  [green]Generated anecdote scenario[/]")
    except Exception as e:
        console.print(f"  [yellow]LLM unavailable ({e}), using fallback prompts[/]")

    try:
        job_id = submit_anecdote_job(
            keyframe_prompts=keyframes,
            motion_prompts=motion_prompts,
            model_size=config.model_size,
            lora_name=config.lora_name,
            gdrive_path=config.gdrive_path,
        )
        console.print(f"  [cyan]Job submitted to Colab:[/] {job_id}")
        console.print(f"  [dim]Waiting for Colab worker to pick it up...[/]")

        anecdote_path = work_dir / "anecdote.mp4"
        download_anecdote_result(job_id, anecdote_path, gdrive_path=config.gdrive_path)
        return anecdote_path, overlay_text

    except Exception as e:
        console.print(f"  [yellow]Colab dispatch failed ({e}), skipping anecdote[/]")
        return None, overlay_text


def _generate_anecdote_via_remote(
    config: PipelineConfig, manifest, work_dir: Path,
) -> tuple[Path | None, str]:
    """Generate the AI anecdote using a remote GPU inference server."""
    from src.anecdote.prompts import (
        ANECDOTE_SYSTEM_PROMPT,
        FALLBACK_KEYFRAMES,
        FALLBACK_MOTION_PROMPTS,
        FALLBACK_OVERLAY,
        build_anecdote_prompt,
        parse_anecdote_response,
    )
    from src.remote import RemoteInferenceClient

    keyframes = list(FALLBACK_KEYFRAMES)
    motion_prompts = list(FALLBACK_MOTION_PROMPTS)
    overlay_text = FALLBACK_OVERLAY

    try:
        import litellm

        user_prompt = build_anecdote_prompt(
            manifest.name,
            manifest.description,
            manifest.project_type.value,
            manifest.readme_content,
        )
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANECDOTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=1000,
        )
        parsed = parse_anecdote_response(response.choices[0].message.content)
        keyframes = parsed["keyframes"]
        motion_prompts = parsed["motion_prompts"]
        overlay_text = parsed.get("overlay_text", FALLBACK_OVERLAY)
        console.print(f"  [green]Generated anecdote scenario[/]")
    except Exception as e:
        console.print(f"  [yellow]LLM unavailable ({e}), using fallback prompts[/]")

    try:
        client = RemoteInferenceClient(config.remote_url)

        console.print("  [dim]Generating keyframes on remote GPU...[/]")
        image_paths = client.generate_keyframes(
            keyframes, work_dir / "keyframes",
        )
        console.print(f"  [green]Got {len(image_paths)} keyframes[/]")

        clip_paths: list[Path] = []
        for i, (img_path, motion) in enumerate(zip(image_paths, motion_prompts)):
            console.print(f"  [dim]Clip {i+1}/{len(image_paths)}:[/] {motion[:50]}...")
            clip_path = work_dir / f"clip_{i:03d}.mp4"
            client.generate_video(img_path, motion, clip_path)
            clip_paths.append(clip_path)

        # Concatenate clips
        anecdote_path = work_dir / "anecdote.mp4"
        if len(clip_paths) == 1:
            import shutil
            shutil.copy2(clip_paths[0], anecdote_path)
        else:
            from src.anecdote.video_gen import _concatenate_clips
            _concatenate_clips(clip_paths, anecdote_path)

        return anecdote_path, overlay_text

    except Exception as e:
        console.print(f"  [yellow]Remote generation failed ({e}), skipping anecdote[/]")
        return None, overlay_text


def _train_lora_via_colab(config: PipelineConfig) -> Path | None:
    """Run LoRA training on Colab via Google Drive."""
    from src.remote import submit_lora_training_job, download_lora_result

    try:
        lora_name = config.lora_name or "custom_style"
        job_id = submit_lora_training_job(
            reference_dir=config.train_lora,
            lora_name=lora_name,
            model_size=config.model_size,
            gdrive_path=config.gdrive_path,
        )
        console.print(f"  [cyan]LoRA training job submitted to Colab:[/] {job_id}")
        console.print(f"  [dim]This may take 15-30 min on an A100...[/]")

        lora_dir = download_lora_result(job_id, lora_name, gdrive_path=config.gdrive_path)
        return lora_dir

    except Exception as e:
        console.print(f"  [yellow]Colab LoRA training failed ({e}), continuing without[/]")
        return None


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, help="Bind port")
@click.option("--model-size", type=click.Choice(["14B", "480P"]), default="480P", help="Wan2.1 I2V resolution")
@click.option("--lora-path", default=None, help="Path to LoRA weights to load")
def serve(host: str, port: int, model_size: str, lora_path: str | None):
    """Start the inference server (for NodeOps / remote GPU hosting)."""
    import os
    os.environ["REPOVIDEO_MODEL_SIZE"] = model_size
    if lora_path:
        os.environ["REPOVIDEO_LORA_PATH"] = lora_path

    try:
        import uvicorn
    except ImportError:
        console.print("[red]Install uvicorn:[/] pip install uvicorn python-multipart")
        sys.exit(1)

    console.print(Panel(
        f"[bold]repovideo serve[/] — Wan2.1 {model_size} inference server\n"
        f"Endpoint: [cyan]http://{host}:{port}[/]"
        + (f"\nLoRA: [cyan]{lora_path}[/]" if lora_path else ""),
        border_style="blue",
    ))

    uvicorn.run("src.serve:app", host=host, port=port, reload=False)


@main.command()
def deploy_info():
    """Show instructions for deploying to NodeOps CreateOS."""
    console.print(Panel(
        "[bold]Deploy repovideo inference to NodeOps CreateOS[/]",
        border_style="blue",
    ))
    console.print("""
[bold]1. Build the Docker image:[/]
   docker build -f Dockerfile.serve -t repovideo-inference .

[bold]2. Push to a registry:[/]
   docker tag repovideo-inference your-registry/repovideo-inference:latest
   docker push your-registry/repovideo-inference:latest

[bold]3. Deploy on NodeOps CreateOS:[/]
   - Go to [cyan]https://createos.nodeops.network/deploy/docker[/]
   - Select Docker Image deployment
   - Enter your image URL
   - Select a GPU instance (A10G for 480P mode, A100 for 14B/720P)
   - Set environment variables:
     [dim]REPOVIDEO_MODEL_SIZE=480P[/]
     [dim]REPOVIDEO_LORA_PATH=/loras/your_style[/]  (optional)
   - Expose port [bold]8000[/]
   - Click Deploy

[bold]4. Upload your LoRA weights:[/]
   After deploy, upload your trained LoRA:
   [dim]curl -X POST https://your-app.nodeops.network/lora/upload \\
     -F "file=@my_lora.zip" -F "name=my_style"[/]

[bold]5. Use from your Mac:[/]
   repovideo generate https://github.com/user/repo \\
     --remote https://your-app.nodeops.network

[bold]Pricing (NodeOps GPU):[/]
   A10G (24 GB, good for 480P): ~$0.60/hr
   A100 (80 GB, needed for 720P): ~$2.50/hr
   Pay-per-second billing, auto-scaling available.
""")


if __name__ == "__main__":
    main()
