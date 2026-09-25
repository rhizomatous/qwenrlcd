from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from qwenrlcd.data import DecisionCollator, DecisionDataset  # noqa: E402
from qwenrlcd.losses import mask_invalid_choices  # noqa: E402
from qwenrlcd.model import DecisionModel, tree_attention_from_labels  # noqa: E402
from qwenrlcd.schema import DecisionBundle, Question  # noqa: E402

from .test_data import WhitespaceTokenizer  # noqa: E402
from .test_schema import example_dict  # noqa: E402


class EchoBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 4)

    def forward(self, *, input_ids, attention_mask, **_kwargs):
        allowed = attention_mask["full_attention"][:, 0] == 0
        weights = allowed.float() / allowed.sum(dim=-1, keepdim=True)
        return SimpleNamespace(
            last_hidden_state=torch.bmm(weights, self.embedding(input_ids))
        )


class SavedBackbone(EchoBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=4)

    def save_pretrained(self, output_dir) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")
        (output_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "hidden_size": 4}), encoding="utf-8"
        )


def test_per_option_head_scores_each_marker_and_normalizes() -> None:
    tokenizer = WhitespaceTokenizer()
    bundle = DecisionBundle.from_dict(example_dict())
    dataset = DecisionDataset(
        [bundle], tokenizer, max_length=512, max_choices=255,
        max_questions=32, shuffle=False, seed=17,
    )
    batch = DecisionCollator(tokenizer, max_choices=255, max_questions=32)([dataset[0]])
    model = DecisionModel(EchoBackbone(), hidden_size=4, max_choices=255)
    logits = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        tree_attention_mask=batch["tree_attention_mask"],
        decision_indices=batch["decision_indices"],
    )
    probabilities = torch.softmax(mask_invalid_choices(logits, batch["num_choices"]), -1)

    assert logits.shape == (1, 3, 3)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones((1, 3)))
    assert probabilities[0, 0, 2] == 0


def test_option_reordering_preserves_key_aligned_scores() -> None:
    tokenizer = WhitespaceTokenizer()
    bundle = DecisionBundle.from_dict(example_dict())
    route = bundle.questions[0]
    reversed_route = Question(
        route.id, route.type, route.instructions,
        tuple(reversed(route.options)), route.target,
    )
    reversed_bundle = DecisionBundle(
        bundle.id, bundle.state, (reversed_route, *bundle.questions[1:]), bundle.source
    )
    dataset = DecisionDataset(
        [bundle, reversed_bundle], tokenizer, max_length=512,
        max_choices=255, max_questions=32, shuffle=False, seed=17,
    )
    batch = DecisionCollator(tokenizer, max_choices=255, max_questions=32)(
        [dataset[0], dataset[1]]
    )
    model = DecisionModel(EchoBackbone(), hidden_size=4, max_choices=255)
    logits = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        tree_attention_mask=batch["tree_attention_mask"],
        decision_indices=batch["decision_indices"],
    )

    assert logits[0, 0, 0] != logits[0, 0, 1]
    torch.testing.assert_close(logits[0, 0, :2], logits[1, 0, :2].flip(0))


def test_compact_labels_reconstruct_dense_tree_with_padding() -> None:
    tokenizer = WhitespaceTokenizer()
    full = DecisionBundle.from_dict(example_dict())
    singleton = DecisionBundle("singleton", full.state, (full.questions[0],))
    dataset = DecisionDataset(
        [full, singleton], tokenizer, max_length=512,
        max_choices=255, max_questions=32, shuffle=False, seed=17,
    )
    examples = [dataset[0], dataset[1]]
    dense = DecisionCollator(tokenizer, 255, 32)(examples)
    compact = DecisionCollator(
        tokenizer, 255, 32, compact_attention_topology=True
    )(examples)

    assert "tree_attention_mask" not in compact
    reconstructed = tree_attention_from_labels(
        compact["token_roles"],
        compact["token_question_indices"],
        compact["token_option_indices"],
    )
    torch.testing.assert_close(reconstructed, dense["tree_attention_mask"])


def test_non_flex_model_accepts_compact_topology() -> None:
    tokenizer = WhitespaceTokenizer()
    bundle = DecisionBundle.from_dict(example_dict())
    dataset = DecisionDataset(
        [bundle], tokenizer, max_length=512,
        max_choices=255, max_questions=32, shuffle=False, seed=17,
    )
    dense = DecisionCollator(tokenizer, 255, 32)([dataset[0]])
    compact = DecisionCollator(
        tokenizer, 255, 32, compact_attention_topology=True
    )([dataset[0]])
    torch.manual_seed(3)
    model = DecisionModel(EchoBackbone(), hidden_size=4, max_choices=255)
    dense_logits = model(
        input_ids=dense["input_ids"],
        position_ids=dense["position_ids"],
        tree_attention_mask=dense["tree_attention_mask"],
        decision_indices=dense["decision_indices"],
    )
    compact_logits = model(
        input_ids=compact["input_ids"],
        position_ids=compact["position_ids"],
        token_roles=compact["token_roles"],
        token_question_indices=compact["token_question_indices"],
        token_option_indices=compact["token_option_indices"],
        decision_indices=compact["decision_indices"],
    )
    torch.testing.assert_close(compact_logits, dense_logits)


def test_full_backbone_components_round_trip(tmp_path, monkeypatch) -> None:
    torch.manual_seed(11)
    original = DecisionModel(
        SavedBackbone(), hidden_size=4, max_choices=17,
        base_model_id="Qwen/Qwen3-1.7B-Base",
    )
    output_dir = tmp_path / "final"
    original.save_components(output_dir)
    saved_config = json.loads(
        (output_dir / "decision_config.json").read_text(encoding="utf-8")
    )
    assert saved_config["backbone_storage"] == "full_model"

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **_kwargs):
            backbone = SavedBackbone()
            backbone.load_state_dict(
                torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            )
            return backbone

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    restored = DecisionModel.load_components(output_dir, dtype=torch.float32)

    assert restored.max_choices == 17
    assert restored.base_model_id == "Qwen/Qwen3-1.7B-Base"
    for expected, actual in zip(
        original.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
