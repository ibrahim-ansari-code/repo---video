"""Load PEFT-saved Wan LoRA weights into diffusers `load_lora_weights`.

`get_peft_model(WanTransformer3DModel).save_pretrained` writes keys like
``base_model.model.blocks.0.attn1.to_q.lora_A.weight``. Diffusers'
``WanTransformer3DModel.load_lora_adapter`` filters on the ``transformer.``
prefix (pipeline-relative), so unconverted checkpoints match nothing and the
LoRA is silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import torch


def convert_peft_wan_lora_state_dict_for_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map PEFT adapter keys to the names ``load_lora_weights`` expects."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model."):
            out["transformer." + key[len("base_model.model.") :]] = value
        elif key.startswith("base_model.blocks."):
            out["transformer." + key[len("base_model.blocks.") :]] = value
        else:
            out[key] = value
    return out


def load_wan_peft_lora_state_dict(lora_path: Path) -> dict[str, torch.Tensor]:
    """Load ``adapter_model.safetensors`` from a PEFT output dir (or a single .safetensors file)."""
    from safetensors.torch import load_file

    if lora_path.is_file():
        if not str(lora_path).endswith(".safetensors"):
            raise ValueError(f"Expected .safetensors file, got {lora_path}")
        raw = load_file(str(lora_path))
    else:
        adapter = lora_path / "adapter_model.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(
                f"No adapter_model.safetensors under {lora_path}. "
                "Expected a PEFT output folder from repovideo LoRA training."
            )
        raw = load_file(str(adapter))
    return convert_peft_wan_lora_state_dict_for_diffusers(raw)
