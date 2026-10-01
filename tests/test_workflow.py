"""Integration coverage using synthetic data, without the local workbook."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from churn_model import clean_data, read_data, training_data, train_models, score_customers, metrics


def sample():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"CustomerID": [f"SYNTHETIC-{i}" for i in range(100)],
                         "tenure": rng.integers(0, 72, 100),
                         "MonthlyCharges": rng.uniform(20, 100, 100),
                         "Contract": ["Month-to-month", "One year"] * 50,
                         "Churn": ["Yes", "No", "No", "Yes"] * 25})


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.features = training_data(sample())
        cls.result = train_models(cls.data, cls.features)

    def test_model_evaluation_and_scoring(self):
        self.assertEqual(self.result["split_sizes"], (60, 20, 20))
        self.assertEqual(len(self.result["comparison"]), 4)
        values = metrics(self.result["test_y"], self.result["test_probability"])
        self.assertTrue(all(0 <= v <= 1 for v in values.values()))
        raw = sample().drop(columns="Churn")
        raw.loc[0, "Contract"] = "Previously unseen"
        scored = score_customers(raw, self.result)
        self.assertEqual(len(scored), 100)
        self.assertTrue(scored.churn_probability.between(0, 1).all())
        self.assertTrue(scored.churn_probability.is_monotonic_decreasing)

    def test_scoring_input_errors(self):
        with self.assertRaisesRegex(ValueError, "Missing prediction"):
            score_customers(sample().drop(columns="Contract"), self.result)
        with self.assertRaisesRegex(ValueError, "no customers"):
            score_customers(sample().iloc[:0], self.result)
        with self.assertRaisesRegex(ValueError, "Threshold"):
            score_customers(sample(), self.result, 1.1)

    def test_model_serialization(self):
        buffer = BytesIO()
        joblib.dump(self.result, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_allclose(score_customers(sample(), restored).churn_probability,
                                   score_customers(sample(), self.result).churn_probability)

    def test_csv_excel_roundtrip(self):
        raw = sample()
        self.assertEqual(read_data(raw.to_csv(index=False).encode(), "input.csv").shape, raw.shape)
        buffer = BytesIO()
        raw.to_excel(buffer, index=False, sheet_name="Customers")
        self.assertEqual(read_data(buffer.getvalue(), "input.xlsx", "Customers").shape, raw.shape)
        self.assertEqual(read_data(raw.to_csv(index=False, sep='\t').encode(), "input.tsv").shape, raw.shape)
        self.assertEqual(read_data(raw.to_csv(index=False, sep=';').encode(), "input.csv").shape, raw.shape)
        self.assertEqual(read_data(raw.to_json(orient='records').encode(), "input.json").shape, raw.shape)
        with self.assertRaisesRegex(ValueError, "CSV"):
            read_data(b"bad", "input.txt")

    def test_duplicate_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            clean_data(pd.DataFrame({"MonthlyCharges": [1], "Monthly Charges": [2]}))

    def test_app_navigation_and_training(self):
        app_path = Path(__file__).resolve().parents[1] / "Customer_prediction.py"
        original_exists, original_read = Path.exists, Path.read_bytes
        def exists(path):
            return True if path.name == "Telco_customer_churn.xlsx" else original_exists(path)
        def read(path):
            return b"synthetic-test-data" if path.name == "Telco_customer_churn.xlsx" else original_read(path)
        with patch.object(Path, "exists", exists), patch.object(Path, "read_bytes", read), patch("churn_model.read_data", return_value=sample()):
            app = AppTest.from_file(str(app_path), default_timeout=90).run()
            self.assertFalse(app.exception)
            app.segmented_control[0].set_value("Local dataset").run()
            self.assertFalse(app.exception)
            app.radio[0].set_value("Model performance").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 4)
            app.slider[0].set_value(0.7).run()
            self.assertFalse(app.exception)
            app.radio[0].set_value("Customer scoring").run()
            app.toggle[0].set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 3)
            app.segmented_control[0].set_value("Upload").run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)

    def test_uploaded_file_automatically_produces_results(self):
        uploaded = BytesIO(sample().to_csv(index=False).encode())
        uploaded.name = "customers.csv"
        with patch("streamlit.file_uploader", return_value=uploaded), patch("churn_model.train_models", wraps=train_models) as train:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Customer_prediction.py"), default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertIn("result", app.session_state)
            self.assertEqual(len(app.metric), 4)
            app.run()
            self.assertEqual(train.call_count, 1)
            uploaded.seek(0)
            uploaded.truncate()
            uploaded.write(b"wrong,data\n1,2")
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertNotIn("result", app.session_state)

    def test_app_without_local_workbook(self):
        original_exists = Path.exists
        def exists(path):
            return False if path.name == "Telco_customer_churn.xlsx" else original_exists(path)
        with patch.object(Path, "exists", exists):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Customer_prediction.py"), default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.info)


if __name__ == "__main__":
    unittest.main()
