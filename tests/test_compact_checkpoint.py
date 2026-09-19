from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
accelerate = pytest.importorskip("accelerate")

from qwenrlcd.compact_checkpoint import (  # noqa: E402
    COMPACT_MODEL_FILE,
    register_compact_model_state_hooks,
)


class TinyDecisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.backbone.requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4)
        self.decision_head = torch.nn.Linear(4, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decision_head(self.adapter(self.backbone(features)))


def prepared_pair(*, seed: int):
    torch.manual_seed(seed)
    accelerator = accelerate.Accelerator(cpu=True)
    model = TinyDecisionModel()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer


@pytest.mark.parametrize("compact", [False, True])
def test_compact_checkpoint_round_trip_and_legacy_load(tmp_path, compact: bool) -> None:
    accelerator, model, optimizer = prepared_pair(seed=17)
    if compact:
        register_compact_model_state_hooks(accelerator)
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    loss = model(features).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = model(features).detach().clone()
    checkpoint = tmp_path / "checkpoint"
    accelerator.save_state(checkpoint)

    assert (checkpoint / COMPACT_MODEL_FILE).exists() == compact
    assert (checkpoint / "model.safetensors").exists() != compact

    restored_accelerator, restored_model, restored_optimizer = prepared_pair(seed=17)
    register_compact_model_state_hooks(restored_accelerator)
    restored_accelerator.load_state(checkpoint)
    torch.testing.assert_close(restored_model(features), expected)
    original_state = optimizer.state_dict()["state"]
    restored_state = restored_optimizer.state_dict()["state"]
    assert original_state.keys() == restored_state.keys()
    assert original_state
    for key in original_state:
        for name, value in original_state[key].items():
            torch.testing.assert_close(restored_state[key][name], value)
