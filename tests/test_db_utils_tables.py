import sys
from pathlib import Path

import sys
from pathlib import Path

import pandas as pd
import pytest
from typing import Dict

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
    mismatch_pred = "pred_ebay_man"
    monkeypatch.setattr(db_utils, "find_hist_tables", lambda: [hist_table])
    monkeypatch.setattr(db_utils, "find_pred_tables", lambda: [pred_table, mismatch_pred])

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

    assert not db_utils.validate_table_consistency(hist_table, mismatch_pred)


def test_load_prediction_data_filters(monkeypatch):
    captured: Dict[str, object] = {}

    def fake_read_sql(query, engine, params=None):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame(
            {
                "date_key": ["2024-01-01"],
                "tyre_fullsize": ["205/55 R16"],
                "tyre_brand": ["B1"],
                "tyre_season_french": ["Ete"],
                "stock_prediction": [1.0],
                "price_prediction": [100.0],
                "ic_price_plus": [110.0],
                "ic_price_minus": [90.0],
                "ic_stock_plus": [1.1],
                "ic_stock_minus": [0.9],
                "prediction_confidence": [0.8],
                "stock_status": ["OK"],
                "volatility_status": ["LOW"],
                "main_rupture_date": ["2024-01-05"],
                "order_recommendation": ["BUY"],
                "tension_days": [5],
                "recommended_volume": [10],
                "optimal_order_date": ["2024-01-02"],
                "last_safe_order_date": ["2024-01-03"],
                "margin_opportunity_days": [2],
                "criticality_score": [0.5],
                "risk_level": ["LOW"],
                "stability_index": [0.9],
                "anomaly_alert": ["NONE"],
                "seasonal_factor": [1.0],
                "supply_chain_alert": ["NONE"],
                "volatility_type": ["NORMAL"],
                "procurement_urgency": ["LOW"],
                "price_jump_alert": [0],
            }
        )

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(db_utils, "prediction_table_exists", lambda name: True)

    df = db_utils.load_prediction_data(
        "tbl",
        brands=["B1", "B2"],
        seasons=["Ete"],
        sizes=["205/55 R16"],
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-31"),
    )

    assert not df.empty
    assert "tbl" in captured["query"]
    assert "tyre_brand IN (:brand0,:brand1)" in captured["query"]
    assert "tyre_season_french IN (:season0)" in captured["query"]
    assert "tyre_fullsize IN (:size0)" in captured["query"]
    assert "date_key >= :start_date" in captured["query"]
    assert "date_key <= :end_date" in captured["query"]
    assert captured["params"]["brand0"] == "B1"
    assert captured["params"]["brand1"] == "B2"
    assert captured["params"]["season0"] == "Ete"
    assert captured["params"]["size0"] == "205/55 R16"
    assert isinstance(df.loc[0, "date_key"], pd.Timestamp)
    assert isinstance(df.loc[0, "main_rupture_date"], pd.Timestamp)
    assert isinstance(df.loc[0, "optimal_order_date"], pd.Timestamp)
    assert isinstance(df.loc[0, "last_safe_order_date"], pd.Timestamp)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
