import unittest

import numpy as np
import pandas as pd

from src.utils.func_model import _prepare_features


class ModelPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        self.other = pd.DataFrame({"a": [4.0], "b": [40.0]})

    def test_linear_model_scales_then_applies_feature_weights(self):
        baseline, _ = _prepare_features(
            self.train, self.other, "LogisticRegression", ["a", "b"]
        )
        weighted, _ = _prepare_features(
            self.train,
            self.other,
            "LogisticRegression",
            ["a", "b"],
            {"a": 2.0},
        )

        np.testing.assert_allclose(weighted[:, 0], baseline[:, 0] * 2.0)
        np.testing.assert_allclose(weighted[:, 1], baseline[:, 1])

    def test_tree_model_keeps_raw_scale(self):
        transformed, _ = _prepare_features(
            self.train, self.other, "RandomForestClassifier", ["a", "b"]
        )
        np.testing.assert_allclose(transformed, self.train.to_numpy())


if __name__ == "__main__":
    unittest.main()
