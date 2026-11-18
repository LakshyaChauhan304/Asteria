"""Tests for additive, non-training-mutating Asteria demo data preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student2_demo_data import DemoDataError, load_demo_data


def _metadata() -> dict[str, object]:
    return {
        "model_name": "demo",
        "model_type": "XGBClassifier",
        "input_features": ["dur", "proto"],
        "number_of_input_features": 2,
        "threshold": 0.5,
        "threshold_status": "Provisional",
    }


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(10, 20)),
            "dur": [float(value) / 10 for value in range(10)],
            "proto": ["tcp", "udp"] * 5,
            "label": [0, 1] * 5,
            "attack_cat": ["Normal", "Fuzzers"] * 5,
        }
    )


class DemoDataTests(unittest.TestCase):
    def _root(self) -> tempfile.TemporaryDirectory[str]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "models").mkdir()
        (root / "data").mkdir()
        (root / "models" / "model_metadata.json").write_text(
            json.dumps(_metadata()), encoding="utf-8"
        )
        _dataset().to_csv(root / "data" / "UNSW_NB15_training-set.csv", index=False)
        _dataset().to_csv(root / "data" / "UNSW_NB15_testing-set.csv", index=False)
        return temporary

    def test_prepares_feature_only_frames_and_preserves_ground_truth_separately(self) -> None:
        with self._root() as temporary:
            bundle = load_demo_data(
                repository_root=Path(temporary), training_rows=6, event_count=3
            )

        self.assertEqual(list(bundle.training_features.columns), ["dur", "proto"])
        self.assertEqual(list(bundle.inference_frame.columns), ["event_id", "dur", "proto"])
        self.assertTrue(all(event_id.startswith("UNSW-") for event_id in bundle.event_ids))
        self.assertNotIn("label", bundle.inference_frame.columns)
        self.assertNotIn("attack_cat", bundle.inference_frame.columns)
        self.assertEqual(set(bundle.ground_truth.columns), {"event_id", "id", "label", "attack_cat"})
        self.assertEqual(bundle.training_row_count, 6)
        self.assertEqual(bundle.missing_feature_diagnostics, ())

    def test_missing_model_feature_fails_closed_with_diagnostic(self) -> None:
        with self._root() as temporary:
            root = Path(temporary)
            broken = _dataset().drop(columns=["proto"])
            broken.to_csv(root / "data" / "UNSW_NB15_training-set.csv", index=False)
            with self.assertRaisesRegex(DemoDataError, r"missing required columns: \['proto'\]"):
                load_demo_data(repository_root=root, training_rows=6, event_count=3)

    def test_event_count_limit_fails_closed(self) -> None:
        with self._root() as temporary, self.assertRaisesRegex(DemoDataError, "between 1 and 100"):
            load_demo_data(repository_root=Path(temporary), training_rows=6, event_count=101)


if __name__ == "__main__":
    unittest.main()
