from __future__ import annotations

from qwenrlcd.data import read_jsonl, write_jsonl
from qwenrlcd.synthetic import generate_records


def test_jsonl_round_trip(tmp_path) -> None:
    records = generate_records(12, seed=9)
    path = tmp_path / "records.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records


def test_synthetic_data_contains_choice_and_noul() -> None:
    records = generate_records(12, seed=3)
    types = {record.question.type.value for record in records}
    assert types == {"choice", "noul"}
