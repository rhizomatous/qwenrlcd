from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .data import (
    TOKEN_ROLE_OPTION,
    TOKEN_ROLE_PADDING,
    TOKEN_ROLE_QUESTION,
    TOKEN_ROLE_STATE,
)


def tree_attention_from_labels(
    token_roles: torch.Tensor,
    token_question_indices: torch.Tensor,
    token_option_indices: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact tree topology represented by compact token labels."""
    if not (
        token_roles.ndim == 2
        and token_roles.shape == token_question_indices.shape == token_option_indices.shape
    ):
        raise ValueError("compact topology tensors must share shape [batch, sequence]")
    sequence_length = token_roles.shape[1]
    positions = torch.arange(sequence_length, device=token_roles.device)
    query_position = positions[None, :, None]
    key_position = positions[None, None, :]
    causal = key_position <= query_position

    query_role = token_roles[:, :, None]
    key_role = token_roles[:, None, :]
    query_question = token_question_indices[:, :, None]
    key_question = token_question_indices[:, None, :]
    query_option = token_option_indices[:, :, None]
    key_option = token_option_indices[:, None, :]
    same_question = query_question == key_question
    same_option = query_option == key_option

    state_query = (
        (query_role == TOKEN_ROLE_STATE)
        & (key_role == TOKEN_ROLE_STATE)
        & causal
    )
    question_query = (query_role == TOKEN_ROLE_QUESTION) & (
        (key_role == TOKEN_ROLE_STATE)
        | ((key_role == TOKEN_ROLE_QUESTION) & same_question & causal)
    )
    option_query = (query_role == TOKEN_ROLE_OPTION) & (
        (key_role == TOKEN_ROLE_STATE)
        | ((key_role == TOKEN_ROLE_QUESTION) & same_question)
        | (
            (key_role == TOKEN_ROLE_OPTION)
            & same_question
            & same_option
            & causal
        )
    )
    padding_query = (
        (query_role == TOKEN_ROLE_PADDING)
        & (key_role == TOKEN_ROLE_PADDING)
        & (key_position == query_position)
    )
    return state_query | question_query | option_query | padding_query


class DecisionModel(nn.Module):
    """Qwen3 with isolated option branches and one shared scalar scorer."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        max_choices: int = 255,
        base_model_id: str | None = None,
        attn_implementation: str = "eager",
        flex_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.max_choices = max_choices
        self.base_model_id = base_model_id
        self.attn_implementation = attn_implementation
        if flex_block_size not in (64, 128):
            raise ValueError("flex_block_size must be 64 or 128")
        self.flex_block_size = flex_block_size
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
        flex_block_size: int = 128,
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
            attn_implementation=attn_implementation,
            flex_block_size=flex_block_size,
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
        flex_block_size: int | None = None,
        is_trainable: bool = False,
    ) -> DecisionModel:
        source = Path(input_dir)
        with (source / "decision_config.json").open(encoding="utf-8") as handle:
            decision_config = json.load(handle)
        if decision_config.get("head_architecture") != "per_option":
            raise ValueError("saved model is not a per-option decision model")

        backbone_dir = source / "backbone"
        storage = decision_config.get("backbone_storage")
        if storage is None:  # Backward compatibility with pre-metadata LoRA exports.
            storage = (
                "peft_adapter"
                if (backbone_dir / "adapter_config.json").is_file()
                else "full_model"
            )
        resolved_flex_block_size = int(
            flex_block_size
            if flex_block_size is not None
            else decision_config.get("flex_block_size", 128)
        )

        if storage == "peft_adapter":
            resolved_model_id = model_id or decision_config.get("base_model_id")
            if not resolved_model_id:
                raise ValueError("base model id is required to reload a PEFT adapter")
            model = cls.from_pretrained(
                resolved_model_id,
                max_choices=int(decision_config["max_choices"]),
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                flex_block_size=resolved_flex_block_size,
            )
            if not (backbone_dir / "adapter_config.json").is_file():
                raise ValueError(f"missing PEFT adapter config in {backbone_dir}")
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(
                model.backbone, backbone_dir, is_trainable=is_trainable
            )
        elif storage == "full_model":
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(
                backbone_dir,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
            config = backbone.config
            if getattr(config, "model_type", None) != "qwen3":
                raise ValueError(
                    "saved full backbone is not a full-attention Qwen3 checkpoint"
                )
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("could not determine saved backbone hidden size")
            model = cls(
                backbone=backbone,
                hidden_size=hidden_size,
                max_choices=int(decision_config["max_choices"]),
                base_model_id=model_id or decision_config.get("base_model_id"),
                attn_implementation=attn_implementation,
                flex_block_size=resolved_flex_block_size,
            )
        else:
            raise ValueError(f"unsupported backbone storage mode: {storage}")

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
        decision_indices: torch.Tensor,
        tree_attention_mask: torch.Tensor | None = None,
        token_roles: torch.Tensor | None = None,
        token_question_indices: torch.Tensor | None = None,
        token_option_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decision_indices.ndim != 3:
            raise ValueError("decision_indices must have shape [batch, questions, choices]")
        compact_tensors = (
            token_roles,
            token_question_indices,
            token_option_indices,
        )
        has_compact_topology = all(tensor is not None for tensor in compact_tensors)
        if not has_compact_topology and any(tensor is not None for tensor in compact_tensors):
            raise ValueError("all three compact topology tensors must be supplied together")
        if tree_attention_mask is None and not has_compact_topology:
            raise ValueError("either a dense tree mask or compact topology labels are required")

        if self.attn_implementation == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            if has_compact_topology:
                assert token_roles is not None
                assert token_question_indices is not None
                assert token_option_indices is not None

                def tree_mask_mod(
                    batch_index: torch.Tensor,
                    _head_index: torch.Tensor,
                    query_index: torch.Tensor,
                    key_index: torch.Tensor,
                ) -> torch.Tensor:
                    query_role = token_roles[batch_index, query_index]
                    key_role = token_roles[batch_index, key_index]
                    same_question = (
                        token_question_indices[batch_index, query_index]
                        == token_question_indices[batch_index, key_index]
                    )
                    same_option = (
                        token_option_indices[batch_index, query_index]
                        == token_option_indices[batch_index, key_index]
                    )
                    causal = key_index <= query_index
                    return (
                        (
                            (query_role == TOKEN_ROLE_STATE)
                            & (key_role == TOKEN_ROLE_STATE)
                            & causal
                        )
                        | (
                            (query_role == TOKEN_ROLE_QUESTION)
                            & (
                                (key_role == TOKEN_ROLE_STATE)
                                | ((key_role == TOKEN_ROLE_QUESTION) & same_question & causal)
                            )
                        )
                        | (
                            (query_role == TOKEN_ROLE_OPTION)
                            & (
                                (key_role == TOKEN_ROLE_STATE)
                                | ((key_role == TOKEN_ROLE_QUESTION) & same_question)
                                | (
                                    (key_role == TOKEN_ROLE_OPTION)
                                    & same_question
                                    & same_option
                                    & causal
                                )
                            )
                        )
                        | (
                            (query_role == TOKEN_ROLE_PADDING)
                            & (key_role == TOKEN_ROLE_PADDING)
                            & (key_index == query_index)
                        )
                    )

                sequence_length = token_roles.shape[-1]
                batch_size = token_roles.shape[0]
                device = token_roles.device
            else:
                assert tree_attention_mask is not None

                def tree_mask_mod(
                    batch_index: torch.Tensor,
                    _head_index: torch.Tensor,
                    query_index: torch.Tensor,
                    key_index: torch.Tensor,
                ) -> torch.Tensor:
                    return tree_attention_mask[batch_index, query_index, key_index]

                sequence_length = tree_attention_mask.shape[-1]
                batch_size = tree_attention_mask.shape[0]
                device = tree_attention_mask.device
            attention_mask = create_block_mask(
                tree_mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=sequence_length,
                KV_LEN=sequence_length,
                device=device,
                BLOCK_SIZE=self.flex_block_size,
                _compile=True,
            )
        else:
            if tree_attention_mask is None:
                assert token_roles is not None
                assert token_question_indices is not None
                assert token_option_indices is not None
                tree_attention_mask = tree_attention_from_labels(
                    token_roles, token_question_indices, token_option_indices
                )
            mask_dtype = next(self.backbone.parameters()).dtype
            attention_mask = torch.zeros(
                (*tree_attention_mask.shape[:1], 1, *tree_attention_mask.shape[1:]),
                dtype=mask_dtype,
                device=tree_attention_mask.device,
            )
            attention_mask.masked_fill_(
                ~tree_attention_mask.unsqueeze(1), torch.finfo(mask_dtype).min
            )

        # Passing the layer-name mapping bypasses Transformers' ordinary triangular
        # mask construction and supplies our shared-prefix/tree topology directly.
        # Dynamic training shapes can route Torch 2.14 through its Flex "decode"
        # lowering even with use_cache=False. Its default 256-row tile is not
        # divisible into our 64/128-row sparse blocks, so pin compatible forward
        # tiles. This changes kernel tiling only; the BlockMask remains exact.
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask={"full_attention": attention_mask},
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **(
                {
                    "kernel_options": {
                        "fwd_BLOCK_M": self.flex_block_size,
                        "fwd_BLOCK_N": 64,
                    }
                }
                if self.attn_implementation == "flex_attention"
                else {}
            ),
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

        if not hasattr(self.backbone, "save_pretrained"):
            raise TypeError("backbone must implement save_pretrained for reloadable exports")
        self.backbone.save_pretrained(destination / "backbone")
        backbone_storage = (
            "peft_adapter" if hasattr(self.backbone, "peft_config") else "full_model"
        )
        torch.save(self.decision_head.state_dict(), destination / "decision_head.pt")
        (destination / "decision_config.json").write_text(
            json.dumps(
                {
                    "max_choices": self.max_choices,
                    "base_model_id": self.base_model_id,
                    "backbone_storage": backbone_storage,
                    "attention_topology": "causal_state_question_option_tree",
                    "head_architecture": "per_option",
                    "flex_block_size": self.flex_block_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
