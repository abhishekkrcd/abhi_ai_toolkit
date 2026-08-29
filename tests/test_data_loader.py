# tests/test_data_loader.py
import unittest
import pandas as pd
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Setup synthetic data that we know the structure of
        data = {
            'feature_A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'feature_B': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target_Y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        self.df = pd.DataFrame(data)
        self.loader = DataLoader("dummy_path.csv", features=['feature_A', 'feature_B'], target='target_Y')

    # tests/test_data_loader.py (Update this method)
    def test_load_and_preprocess_success(self):
        # Setup the synthetic data frame to ensure we pass valid input
        data = {
            'feature_A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'feature_B': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target_Y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        input_df = pd.DataFrame(data)

        # 1. Test Step: Call the method
        # This will fail because the method does not return the required final data structure.
        X, y = self.loader.preprocess(input_df)

        # 2. Assert the failure: We expect X and y to be non-empty,
        # but since the actual method is empty, we assert they are empty/fail to meet a minimal size.
        # The failure will be: Assertion Error: Expected data structure with min size > 0.
        self.assertGreater(X.shape[0], 0, "Processed features X should not be empty.")