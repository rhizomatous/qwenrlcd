from __future__ import annotations

import json

from qwenrlcd.hf_data import HFDatasetBundles, bundle_from_dataset_row


def test_hf_row_uses_decisions_and_source_without_provenance_details() -> None:
    row = {
        "schema_version": "qwenrlcd.bundle.v1",
        "id": "example",
        "state_json": json.dumps({"message": "hello"}),
        "questions": [{
            "id": "sentiment",
            "type": "choice",
            "instructions_json": json.dumps("What is the sentiment?"),
            "options": [
                {"key": "negative", "description_json": json.dumps("negative")},
                {"key": "positive", "description_json": json.dumps("positive")},
            ],
            "target": [0.25, 0.75],
        }],
        "provenance": {"source": "fixture", "details_json": "not parsed"},
    }
    bundle = bundle_from_dataset_row(row)
    assert bundle.source == "fixture"
    assert bundle.state == {"message": "hello"}
    assert bundle.questions[0].target_vector() == [0.25, 0.75]
    assert HFDatasetBundles([row])[0] == bundle
    assert len(HFDatasetBundles([row])) == 1
