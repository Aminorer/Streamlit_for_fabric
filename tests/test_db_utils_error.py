import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_load_hist_data_returns_empty_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(db_utils, "find_hist_tables", lambda: ["tbl"])
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", {"tbl"})
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        df = db_utils.load_hist_data()
    assert df.empty
    assert "Erreur lors du chargement des données historiques" in caplog.text


def test_load_prediction_data_returns_empty_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine, params=None):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(db_utils, "find_pred_tables", lambda: ["tbl"])
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", {"tbl"})
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        df = db_utils.load_prediction_data()
    assert df.empty
    assert "Erreur lors du chargement des données de prédiction" in caplog.text


def test_save_dataframe_to_table_handles_error(monkeypatch, caplog):
    df = pd.DataFrame({"a": [1]})

    def fake_to_sql(self, *args, **kwargs):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    monkeypatch.setattr(db_utils, "ALLOWED_TABLES", {"tbl"})

    with caplog.at_level("ERROR"):
        result = db_utils.save_dataframe_to_table(df, "tbl")
    assert result.empty
    assert "Erreur lors de l'enregistrement de la table" in caplog.text


def test_find_hist_tables_handles_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        tables = db_utils.find_hist_tables()
    assert tables == []
    assert "Erreur lors de la récupération des tables historiques" in caplog.text


def test_find_pred_tables_handles_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        tables = db_utils.find_pred_tables()
    assert tables == []
    assert "Erreur lors de la récupération des tables de prédiction" in caplog.text
