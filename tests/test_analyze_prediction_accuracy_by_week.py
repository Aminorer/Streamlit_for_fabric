import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Dynamically load the module since the filename starts with a digit
spec = importlib.util.spec_from_file_location(
    "analyse_module", Path("pages") / "1_Analyse_Comparative.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

analyze_prediction_accuracy_by_week = module.analyze_prediction_accuracy_by_week


def test_weekly_accuracy():
    hist_df = pd.DataFrame(
        {
            "date_key": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-08",
                "2024-01-09",
            ],
            "Sum_stock_quantity": [10, 10, 10, 20],
        }
    )
    pred_df = pd.DataFrame(
        {
            "date_key": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-08",
                "2024-01-09",
            ],
            "stock_prediction": [8, 12, 5, 25],
        }
    )
    result = analyze_prediction_accuracy_by_week(hist_df, pred_df)
    assert result["week"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
    ]
    first_week = result.loc[result["week"] == pd.Timestamp("2024-01-01"), "accuracy"].iloc[0]
    second_week = result.loc[result["week"] == pd.Timestamp("2024-01-08"), "accuracy"].iloc[0]
    assert pytest.approx(first_week, rel=1e-6) == 0.8
    assert pytest.approx(second_week, rel=1e-6) == 0.625


def test_excludes_zero_stock_rows():
    hist_df = pd.DataFrame(
        {
            "date_key": ["2024-01-02", "2024-01-09"],
            "Sum_stock_quantity": [0, 5],
        }
    )
    pred_df = pd.DataFrame(
        {
            "date_key": ["2024-01-02", "2024-01-09"],
            "stock_prediction": [3, 6],
        }
    )
    result = analyze_prediction_accuracy_by_week(hist_df, pred_df)
    assert result["week"].tolist() == [pd.Timestamp("2024-01-08")]
    assert pytest.approx(result.loc[0, "accuracy"], rel=1e-6) == 0.8


def test_all_zero_stock_returns_empty():
    hist_df = pd.DataFrame({"date_key": ["2024-01-02"], "Sum_stock_quantity": [0]})
    pred_df = pd.DataFrame({"date_key": ["2024-01-02"], "stock_prediction": [5]})
    result = analyze_prediction_accuracy_by_week(hist_df, pred_df)
    assert result.empty
