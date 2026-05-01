from __future__ import annotations

import unittest

import numpy as np

from football_analysis.data import (
    add_targets,
    build_model_frame,
    load_raw_data,
    map_position_target,
    map_role_target,
)
from football_analysis.modeling import build_preprocessor


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_data = load_raw_data()

    def test_targets_are_deterministic(self) -> None:
        first = add_targets(self.raw_data)
        second = add_targets(self.raw_data)
        self.assertTrue(first["role_target"].equals(second["role_target"]))
        self.assertTrue(first["position_target"].equals(second["position_target"]))

    def test_target_mapping_examples(self) -> None:
        self.assertEqual(map_role_target("ST"), "attack")
        self.assertEqual(map_role_target("CDM"), "midfield")
        self.assertEqual(map_role_target("CB"), "defense")
        self.assertEqual(map_role_target("GK"), "goalkeeper")
        self.assertEqual(map_position_target("CF"), "ST")
        self.assertEqual(map_position_target("RWB"), "RB")
        self.assertEqual(map_position_target("LWB"), "LB")

    def test_preprocessor_output_is_numeric(self) -> None:
        frame, numeric_features, categorical_features, _ = build_model_frame(
            self.raw_data, "role"
        )
        sample = frame.head(256)
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(
            sample[numeric_features + categorical_features]
        )
        self.assertTrue(np.issubdtype(transformed.dtype, np.number))


if __name__ == "__main__":
    unittest.main()
