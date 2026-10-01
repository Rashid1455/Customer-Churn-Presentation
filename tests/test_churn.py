import unittest
import pandas as pd
from churn_model import clean_data, training_data, FEATURES


class DataTests(unittest.TestCase):
    def example(self):
        return pd.DataFrame({'CustomerID': [f'C{i}' for i in range(40)],
                             'Churn': ['Yes', 'No'] * 20,
                             'tenure': list(range(40)), 'MonthlyCharges': [45] * 40,
                             'Churn Score': [99] * 40, 'Churn Reason': ['Outcome'] * 40})

    def test_outcome_leakage_excluded(self):
        df, features = training_data(self.example())
        self.assertEqual(features, ['tenure_months', 'monthly_charges'])
        self.assertEqual(df.churn_value.sum(), 20)

    def test_invalid_target_rejected(self):
        raw = self.example()
        raw.loc[0, 'Churn'] = 'Unknown'
        with self.assertRaisesRegex(ValueError, 'valid churn label'):
            training_data(raw)

    def test_customer_overlap_rejected(self):
        raw = self.example()
        raw.loc[0, 'CustomerID'] = 'C1'
        with self.assertRaisesRegex(ValueError, 'one row per customer'):
            training_data(raw)

    def test_blank_charges(self):
        df = clean_data(pd.DataFrame({'TotalCharges': [' ', ' '], 'tenure': [0, 3]}))
        self.assertEqual(df.total_charges.iloc[0], 0)
        self.assertTrue(pd.isna(df.total_charges.iloc[1]))


if __name__ == '__main__':
    unittest.main()
