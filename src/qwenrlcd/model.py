from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


class DecisionModel(nn.Module):
    """Qwen3 with isolated option branches and one shared scalar scorer."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        max_choices: int = 255,
        base_model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.max_choices = max_choices
        self.base_model_id = base_model_id
        self.decision_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        max_choices: int = 255,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        attn_implementation: str = "eager",
    ) -> DecisionModel:
        from transformers import AutoModelForCausalLM

        causal_lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        backbone = causal_lm.base_model
        if backbone is causal_lm and hasattr(causal_lm, "model"):
            backbone = causal_lm.model

        config = getattr(backbone, "config", causal_lm.config)
        if getattr(config, "model_type", None) != "qwen3":
            raise ValueError(
                "packed tree attention currently requires a full-attention Qwen3 checkpoint"
            )
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("could not determine backbone hidden size")

        model = cls(
            backbone=backbone,
            hidden_size=hidden_size,
            max_choices=max_choices,
            base_model_id=model_id,
        )
        del causal_lm
        return model

    @classmethod
    def load_components(
        cls,
        input_dir: str | Path,
        *,
        model_id: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        attn_implementation: str = "eager",
        is_trainable: bool = False,
    ) -> DecisionModel:
        source = Path(input_dir)
        with (source / "decision_config.json").open(encoding="utf-8") as handle:
            decision_config = json.load(handle)
        if decision_config.get("head_architecture") != "per_option":
            raise ValueError("saved model is not a per-option decision model")

        resolved_model_id = model_id or decision_config.get("base_model_id")
        if not resolved_model_id:
            raise ValueError("base model id is required to reload saved components")

        model = cls.from_pretrained(
            resolved_model_id,
            max_choices=int(decision_config["max_choices"]),
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        adapter_dir = source / "backbone"
        if not (adapter_dir / "adapter_config.json").is_file():
            raise ValueError(
                f"{adapter_dir} is not a saved PEFT adapter; full-backbone reload "
                "is not implemented"
            )
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(
            model.backbone, adapter_dir, is_trainable=is_trainable
        )
        head_state = torch.load(
            source / "decision_head.pt", map_location="cpu", weights_only=True
        )
        model.decision_head.load_state_dict(head_state)
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
        position_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        decision_indices: torch.Tensor,
    ) -> torch.Tensor:
        if decision_indices.ndim != 3:
            raise ValueError("decision_indices must have shape [batch, questions, choices]")
        mask_dtype = next(self.backbone.parameters()).dtype
        additive_mask = torch.zeros(
            (*tree_attention_mask.shape[:1], 1, *tree_attention_mask.shape[1:]),
            dtype=mask_dtype,
            device=tree_attention_mask.device,
        )
        additive_mask.masked_fill_(
            ~tree_attention_mask.unsqueeze(1), torch.finfo(mask_dtype).min
        )

        # Passing the layer-name mapping bypasses Transformers' ordinary triangular
        # mask construction and supplies our shared-prefix/tree topology directly.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask={"full_attention": additive_mask},
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        batch_rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        batch_rows = batch_rows[:, None, None].expand_as(decision_indices)
        pooled = hidden_states[batch_rows, decision_indices]
        pooled = pooled.to(dtype=self.decision_head.weight.dtype)
        return self.decision_head(pooled).squeeze(-1)

    def save_components(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(destination / "backbone")
        else:
            torch.save(self.backbone.state_dict(), destination / "backbone.pt")
        torch.save(self.decision_head.state_dict(), destination / "decision_head.pt")
        (destination / "decision_config.json").write_text(
            json.dumps(
                {
                    "max_choices": self.max_choices,
                    "base_model_id": self.base_model_id,
                    "attention_topology": "causal_state_question_option_tree",
                    "head_architecture": "per_option",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
