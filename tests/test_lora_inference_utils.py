"""PEFT → diffusers Wan LoRA key rewriting."""

import torch

from src.anecdote.lora_inference_utils import convert_peft_wan_lora_state_dict_for_diffusers


def test_convert_peft_prefix_to_transformer():
    peft = {
        "base_model.model.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.blocks.0.attn1.to_q.lora_B.weight": torch.zeros(2, 2),
    }
    out = convert_peft_wan_lora_state_dict_for_diffusers(peft)
    assert "transformer.blocks.0.attn1.to_q.lora_A.weight" in out
    assert "base_model.model." not in "".join(out.keys())


def test_passthrough_diffusers_style_keys():
    sd = {"transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(1, 1)}
    out = convert_peft_wan_lora_state_dict_for_diffusers(sd)
    assert out == sd
