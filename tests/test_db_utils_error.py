import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_load_hist_data_raises_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(db_utils, "find_hist_tables", lambda: ["tbl"])
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        with pytest.raises(SQLAlchemyError):
            db_utils.load_hist_data()
    assert "Erreur lors du chargement des données historiques" in caplog.text


def test_load_prediction_data_raises_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine, params=None):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(db_utils, "prediction_table_exists", lambda name: True)
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        with pytest.raises(SQLAlchemyError):
            db_utils.load_prediction_data("tbl")
    assert "Erreur lors du chargement des données de prédiction" in caplog.text


def test_save_dataframe_to_table_raises_on_error(monkeypatch, caplog):
    df = pd.DataFrame({"a": [1]})

    def fake_to_sql(self, *args, **kwargs):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    with caplog.at_level("ERROR"):
        with pytest.raises(SQLAlchemyError):
            db_utils.save_dataframe_to_table(df, "tbl")
    assert "Erreur lors de l'enregistrement de la table" in caplog.text


def test_find_hist_tables_raises_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_hist", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        with pytest.raises(SQLAlchemyError):
            db_utils.find_hist_tables()
    assert "Erreur lors de la récupération des tables historiques" in caplog.text


def test_find_pred_tables_raises_on_error(monkeypatch, caplog):
    def fake_read_sql(query, engine):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(db_utils, "get_engine_pred", lambda: object())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    with caplog.at_level("ERROR"):
        with pytest.raises(SQLAlchemyError):
            db_utils.find_pred_tables()
    assert "Erreur lors de la récupération des tables de prédiction" in caplog.text
