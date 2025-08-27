import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_load_prediction_data_invalid_table_name():
    with pytest.raises(ValueError):
        db_utils.load_prediction_data("bad;table")


def test_load_prediction_data_missing_table(monkeypatch):
    monkeypatch.setattr(db_utils, "prediction_table_exists", lambda name: False)
    with pytest.raises(ValueError):
        db_utils.load_prediction_data("pred_missing")


def test_save_dataframe_to_table_invalid_name():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        db_utils.save_dataframe_to_table(df, "bad name")
