import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_find_tables_via_columns(monkeypatch):
    def fake_read_sql(query, engine, params=None):
        if "INFORMATION_SCHEMA.TABLES" in query:
            return pd.DataFrame({"TABLE_NAME": ["hist_tbl", "pred_tbl", "other_tbl"]})
        elif "INFORMATION_SCHEMA.COLUMNS" in query:
            table = params["table"]
            if table == "hist_tbl":
                return pd.DataFrame({"COLUMN_NAME": ["Sum_stock_quantity", "foo"]})
            if table == "pred_tbl":
                return pd.DataFrame({"COLUMN_NAME": ["stock_prediction", "bar"]})
            return pd.DataFrame({"COLUMN_NAME": ["foo"]})
        raise AssertionError("Unexpected query")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    assert db_utils.find_hist_tables() == ["hist_tbl"]
    assert db_utils.find_pred_tables() == ["pred_tbl"]


def test_find_tables_case_insensitive(monkeypatch):
    def fake_read_sql(query, engine, params=None):
        if "INFORMATION_SCHEMA.TABLES" in query:
            return pd.DataFrame({"TABLE_NAME": ["hist_tbl", "pred_tbl"]})
        elif "INFORMATION_SCHEMA.COLUMNS" in query:
            table = params["table"]
            if table == "hist_tbl":
                return pd.DataFrame({"COLUMN_NAME": ["SUM_stock_QUANTITY"]})
            if table == "pred_tbl":
                return pd.DataFrame({"COLUMN_NAME": ["STOCK_PREDICTION"]})
        raise AssertionError("Unexpected query")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    assert db_utils.find_hist_tables() == ["hist_tbl"]
    assert db_utils.find_pred_tables() == ["pred_tbl"]
