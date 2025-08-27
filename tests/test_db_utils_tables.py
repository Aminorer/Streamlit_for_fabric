import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_discover_platforms(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_hist_tables",
        lambda: [
            "fullsize_stock_hist_amz_man",
            "fullsize_stock_hist_ebay_dis",
        ],
    )
    monkeypatch.setattr(
        db_utils,
        "find_pred_tables",
        lambda: [
            "pred_amz_man",
            "pred_amz_dis",
            "pred_ebay_dis",
        ],
    )

    expected = {"amz": ["dis", "man"], "ebay": ["dis"]}
    assert db_utils.discover_platforms() == expected


def test_get_matching_tables(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_hist_tables",
        lambda: ["fullsize_stock_hist_amz_man", "fullsize_stock_hist_ebay_dis"],
    )
    monkeypatch.setattr(
        db_utils,
        "find_pred_tables",
        lambda: ["pred_amz_man", "pred_ebay_dis"],
    )

    hist, pred = db_utils.get_matching_tables("amz", "man")
    assert hist == "fullsize_stock_hist_amz_man"
    assert pred == "pred_amz_man"


def test_validate_table_consistency(monkeypatch, caplog):
    hist_table = "fullsize_stock_hist_amz_man"
    pred_table = "pred_amz_man"
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", {hist_table, pred_table, "pred_ebay_man"})

    def fake_read_sql(query, engine):
        if "fullsize_stock_hist" in query:
            return pd.DataFrame(columns=["a", "b"])
        return pd.DataFrame(columns=["a", "b", "c"])

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    assert db_utils.validate_table_consistency(hist_table, pred_table)

    bad_pred = "pred_amz_man"

    def fake_read_sql_bad(query, engine):
        if "fullsize_stock_hist" in query:
            return pd.DataFrame(columns=["a", "b"])
        return pd.DataFrame(columns=["a"])

    monkeypatch.setattr(pd, "read_sql", fake_read_sql_bad)
    with caplog.at_level("ERROR"):
        assert not db_utils.validate_table_consistency(hist_table, bad_pred)
        assert "Incompatible schemas" in caplog.text

    mismatch_pred = "pred_ebay_man"
    assert not db_utils.validate_table_consistency(hist_table, mismatch_pred)
