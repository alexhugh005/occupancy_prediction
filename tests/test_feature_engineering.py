"""Basic unit tests for feature engineering."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from feature_engineering import FeatureEngineer, parse_datetime


def test_parse_datetime():
    df = pd.DataFrame({"date": ["2015-02-04 17:51:00"]})
    out = parse_datetime(df)
    assert "hour" in out.columns
    assert out["hour"].iloc[0] == 17
    assert out["dayofweek"].iloc[0] == 2  # Wednesday
    assert out["time_of_day"].iloc[0] == "afternoon"


def test_feature_engineer_fit_transform():
    df = pd.DataFrame({
        "date": ["2015-02-04 17:51:00", "2015-02-05 09:00:00"],
        "Temperature": [23.0, 21.0],
        "Humidity": [27.0, 30.0],
        "Light": [400.0, 0.0],
        "CO2": [700.0, 450.0],
        "HumidityRatio": [0.0048, 0.0045],
        "Occupancy": [1, 0],
    })
    fe = FeatureEngineer()
    X = fe.fit_transform(df)
    assert X.shape[0] == 2
    assert X.shape[1] > 5
    # Transform only should work after fit
    X2 = fe.transform(df)
    assert list(X.columns) == list(X2.columns)
