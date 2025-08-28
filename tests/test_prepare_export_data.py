import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_prepare_export_data_columns_and_rows():
    df_hist = pd.DataFrame(
        {
            "date_key": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "tyre_brand": ["A", "B"],
            "tyre_season_french": ["ETE", "HIVER"],
            "tyre_fullsize": ["205/55R16", "195/65R15"],
            "Sum_stock_quantity": [10, 20],
        }
    )

    week1 = pd.DataFrame(
        {
            "date_key": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "tyre_brand": ["A", "B"],
            "tyre_season_french": ["ETE", "HIVER"],
            "tyre_fullsize": ["205/55R16", "195/65R15"],
            "stock_prediction": [11.0, 19.0],
        }
    )

    week2 = pd.DataFrame(
        {
            "date_key": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "tyre_brand": ["A", "B"],
            "tyre_season_french": ["ETE", "HIVER"],
            "tyre_fullsize": ["205/55R16", "195/65R15"],
            "stock_prediction": [12.0, 18.0],
        }
    )

    predictions_dict = {
        "Semaine 1": week1,
        "Semaine 2": week2,
    }

    result = db_utils.prepare_export_data(df_hist, predictions_dict)

    # Verify expected columns exist
    assert {
        "prediction_week",
        "stock_prediction",
        "Sum_stock_quantity",
    }.issubset(result.columns)

    # Expect two weeks * two rows each = four rows total
    assert len(result) == 4
