import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_aggregate_predictions_missing_confidence_columns():
    df = pd.DataFrame(
        {
            "date_key": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "tyre_brand": ["A", "A"],
            "stock_prediction": [1.0, 2.0],
        }
    )
    result = db_utils.aggregate_predictions(df, show_confidence=True)
    assert list(result.columns) == ["date_key", "tyre_brand", "stock_prediction"]
    assert result.loc[0, "stock_prediction"] == 3.0


def test_aggregate_predictions_with_confidence_columns():
    df = pd.DataFrame(
        {
            "date_key": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "tyre_brand": ["A", "A"],
            "stock_prediction": [1.0, 2.0],
            "ic_stock_plus": [1.1, 2.1],
            "ic_stock_minus": [0.9, 1.9],
        }
    )
    result = db_utils.aggregate_predictions(df, show_confidence=True)
    assert "ic_stock_plus" in result.columns
    assert "ic_stock_minus" in result.columns
    assert result.loc[0, "ic_stock_plus"] == pytest.approx(3.2)
    assert result.loc[0, "ic_stock_minus"] == pytest.approx(2.8)
