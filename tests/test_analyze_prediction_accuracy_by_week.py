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


def test_excludes_zero_stock_rows():
    hist_df = pd.DataFrame(
        {
            "date_key": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Sum_stock_quantity": [10, 0, 5],
        }
    )
    pred_df = pd.DataFrame(
        {
            "date_key": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "stock_prediction": [8, 1, 6],
        }
    )
    result = analyze_prediction_accuracy_by_week(hist_df, pred_df)
    assert not result.empty
    assert pytest.approx(result.loc[0, "accuracy"], rel=1e-6) == 0.8


def test_all_zero_stock_returns_empty():
    hist_df = pd.DataFrame({"date_key": ["2024-01-01"], "Sum_stock_quantity": [0]})
    pred_df = pd.DataFrame({"date_key": ["2024-01-01"], "stock_prediction": [5]})
    result = analyze_prediction_accuracy_by_week(hist_df, pred_df)
    assert result.empty
