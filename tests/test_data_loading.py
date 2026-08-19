import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.func_common import load_data_df


class LoadDataDfTests(unittest.TestCase):
    def test_loads_exact_dataset_name_and_preprocesses_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "DATE": ["2024-01-01 12:30", "2024-01-02 13:30"],
                    "sensor": [1.0, None],
                    "TAG": ["OK", "NG"],
                }
            ).to_csv(data_dir / "sample.csv", index=False)
            pd.DataFrame({"DATE": [], "TAG": []}).to_csv(
                data_dir / "sample-backup.csv", index=False
            )

            df, ng_df, target_cols, check_cols, date_cols = load_data_df(
                "sample", data_dir
            )

            self.assertEqual(target_cols, ["TAG"])
            self.assertEqual(check_cols, ["sensor"])
            self.assertEqual(date_cols, ["DATE"])
            self.assertEqual(df["TAG"].tolist(), [0, 1])
            self.assertEqual(len(ng_df), 1)
            self.assertFalse(df["sensor"].isna().any())
            self.assertTrue((df["DATE"].dt.hour == 0).all())

    def test_missing_dataset_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset 'missing'"):
                load_data_df("missing", temp_dir)

    def test_rejects_non_numeric_feature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame(
                {"DATE": ["2024-01-01"], "category": ["A"], "TAG": ["OK"]}
            ).to_csv(data_dir / "sample.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Non-numeric columns: category"):
                load_data_df("sample", data_dir)


if __name__ == "__main__":
    unittest.main()
