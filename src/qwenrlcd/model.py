from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, max_choices: int = 255) -> None:
        super().__init__()
        self.backbone = backbone
        self.max_choices = max_choices
        self.decision_head = nn.Linear(hidden_size, max_choices)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        max_choices: int = 255,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> DecisionModel:
        from transformers import AutoModelForCausalLM

        # Transformers unwraps Qwen3.5's composite checkpoint into its text-only
        # Qwen3_5ForCausalLM class. This avoids loading the unused vision tower.
        causal_lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        backbone = causal_lm.base_model
        if backbone is causal_lm and hasattr(causal_lm, "model"):
            backbone = causal_lm.model

        config = getattr(backbone, "config", causal_lm.config)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError("could not determine backbone hidden size")

        model = cls(backbone=backbone, hidden_size=hidden_size, max_choices=max_choices)
        del causal_lm
        return model

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def attach_lora(
        self,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: Sequence[str],
    ) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, config)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decision_indices: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs.hidden_states[-1]

        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        pooled = hidden_states[rows, decision_indices]
        pooled = pooled.to(dtype=self.decision_head.weight.dtype)
        return self.decision_head(pooled)

    def save_components(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(destination / "backbone")
        else:
            torch.save(self.backbone.state_dict(), destination / "backbone.pt")
        torch.save(self.decision_head.state_dict(), destination / "decision_head.pt")
        (destination / "decision_config.json").write_text(
            json.dumps({"max_choices": self.max_choices}, indent=2) + "\n",
            encoding="utf-8",
        )
