import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_find_hist_tables_respects_whitelist(monkeypatch):
    db_utils.find_hist_tables.clear()
    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())

    def fake_read_sql(query, engine):
        return pd.DataFrame({"TABLE_NAME": ["fullsize_stock_hist_a", "fullsize_stock_hist_b"]})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", {"fullsize_stock_hist_b"})

    assert db_utils.find_hist_tables() == ["fullsize_stock_hist_b"]


def test_save_dataframe_to_table_disallowed(monkeypatch):
    df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", set())

    with pytest.raises(ValueError):
        db_utils.save_dataframe_to_table(df, "tbl")
