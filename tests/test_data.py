import pandas as pd

from src.data.clean import clean_dataframe
from src.data.feature_engineer import engineer_features
from src.data.merge import merge_dataframes


def test_clean_dataframe_standardizes_columns():
    df = pd.DataFrame(
        [
            {"Temperature ": 20, "Humidity": 50, "Date": "2024-01-01"},
            {"Temperature ": 20, "Humidity": None, "Date": "2024-01-01"},
        ]
    )

    cleaned = clean_dataframe(df)
    assert "temperature" in cleaned.columns
    assert "humidity" in cleaned.columns
    assert cleaned["humidity"].iloc[0] == cleaned["humidity"].iloc[1]
    assert cleaned["date"].dtype == "datetime64[ns]"


def test_merge_dataframes_requires_keys():
    df_a = pd.DataFrame(
        [
            {"date": "2024-01-01", "location": "Nairobi", "cases": 5},
            {"date": "2024-01-02", "location": "Kisumu", "cases": 3},
        ]
    )
    df_b = pd.DataFrame(
        [
            {"date": "2024-01-01", "location": "Nairobi", "humidity": 60},
            {"date": "2024-01-02", "location": "Kisumu", "humidity": 55},
        ]
    )

    merged = merge_dataframes([df_a, df_b], merge_keys=["date", "location"])
    assert "cases" in merged.columns
    assert "humidity" in merged.columns
    assert merged.shape[0] == 2


def test_engineer_features_adds_derived_columns():
    df = pd.DataFrame(
        {
            "temperature": [25, 26],
            "humidity": [65, 70],
            "cases": [10, 12],
            "precipitation": [5.0, 0.0],
            "affected_population": [1000, 1000],
        }
    )
    features = engineer_features(df, rolling_window=2)
    assert "temp_humidity_index" in features.columns
    assert "rain_to_case_ratio" in features.columns
    assert "cases_roll_mean_2" in features.columns
    assert features["impact_ratio"].iloc[0] == 0.01
