"""
Feature Engineering module for Occupancy Prediction.
Extracts datetime features and applies one-hot encoding as specified.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List
import joblib
import os


def parse_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parse date column and extract temporal features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["hour"] = df[date_col].dt.hour
    df["dayofweek"] = df[date_col].dt.dayofweek  # 0=Mon ... 6=Sun
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Time of day bins (categorical)
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
    )
    return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature column names."""
    numeric = [
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "hour",
        "day",
        "month",
        "is_weekend",
    ]
    categorical = ["dayofweek", "time_of_day"]
    return numeric, categorical


class FeatureEngineer:
    """Handles feature engineering including one-hot encoding of categoricals."""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        df = parse_datetime(df)
        self.numeric_cols, self.categorical_cols = get_feature_columns()
        # Ensure columns exist
        for col in self.categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Missing categorical column: {col}")
        cat_data = df[self.categorical_cols].astype(str)
        self.encoder.fit(cat_data)
        # Build feature names
        ohe_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names_ = self.numeric_cols + ohe_names
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        df = parse_datetime(df)
        numeric = df[self.numeric_cols].reset_index(drop=True)
        cat_data = df[self.categorical_cols].astype(str)
        ohe = pd.DataFrame(
            self.encoder.transform(cat_data),
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
        )
        features = pd.concat([numeric, ohe], axis=1)
        # Ensure consistent column order
        features = features.reindex(columns=self.feature_names_, fill_value=0.0)
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        return joblib.load(path)
