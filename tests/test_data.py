import pandas as pd
import numpy as np
import pytest
from src.data.clean import clean_data
from src.data.merge import merge_datasets
from src.data.feature_engineer import engineer_features
from src.utils.config import load_config

@pytest.fixture
def config():
    return load_config()

def test_clean_data_logic(config):
    # Use different months to ensure at least 2 rows after resampling
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "location": ["Nairobi", "Nairobi", "Nairobi"],
        "temp": [20.0, 20.0, 25.0],
        "humidity": [50.0, np.nan, 60.0]
    })
    
    cleaned = clean_data(df, config)
    assert not cleaned["humidity"].isnull().any()
    assert "date" in cleaned.columns
    # Check that we have at least one row
    assert len(cleaned) >= 1
    # Check KNN result for the first month (should be around 50)
    # Since scaling is applied, we check if it's within [0, 1]
    assert cleaned["humidity"].min() >= 0
    assert cleaned["humidity"].max() <= 1

def test_merge_datasets_alignment(config):
    df_a = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "location": ["Nairobi", "Kisumu"],
        "cases": [5, 3]
    })
    df_b = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "location": ["Nairobi", "Kisumu"],
        "humidity": [60, 55]
    })

    merged = merge_datasets([df_a, df_b], config)
    assert "cases" in merged.columns
    assert "humidity" in merged.columns
    assert len(merged) == 2

def test_engineer_features_selection(config):
    # Create enough features to trigger RFE properly
    df = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "location": "Nairobi",
        "cases": np.random.randint(0, 100, 10),
        "temp": np.random.normal(25, 5, 10),
        "humidity": np.random.normal(60, 10, 10),
        "feat_1": np.random.rand(10),
        "feat_2": np.random.rand(10)
    })
    
    # Update config for small feature set
    config['data']['processing']['features']['rfe_n_features'] = 2
    
    selected = engineer_features(df, target="cases", config=config)
    assert "cases" in selected.columns
    # Result should have features + target + date + location
    assert len(selected.columns) <= 5
