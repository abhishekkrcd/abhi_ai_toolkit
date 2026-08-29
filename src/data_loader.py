# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """Handles loading, preprocessing, and splitting of the dataset."""
    def __init__(self, data_path: str, features: list, target: str):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.X = None
        self.y = None

    def load_data(self) -> pd.DataFrame:
        """Simulates loading data from a CSV."""
        print(f"Loading data from {self.data_path}...")
        # NOTE: For this quest, we'll create a synthetic dataset in the next step!
        return pd.DataFrame()

    def preprocess(self, df: pd.DataFrame):
        """Extracts features and target."""
        print("Preprocessing features and target...")
        # Placeholder: Real logic would handle normalization, encoding, etc.
        # For now, we just split the columns.
        self.X = df[self.features]
        self.y = df[self.target]
        return self.X, self.y

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Splits data into training and testing sets."""
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded and preprocessed before splitting.")
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)