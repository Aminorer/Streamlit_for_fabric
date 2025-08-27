import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_run_diagnostics_success():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO t1 (name) VALUES ('a'), ('b')"))

    report = db_utils.run_diagnostics(engine)
    assert report["connected"] is True
    assert "t1" in report["tables"]
    assert report["tables"]["t1"] == ["id", "name"]
    assert "t1" in report["samples"]
    assert not report["samples"]["t1"].empty
    assert report["errors"] == []


def test_run_diagnostics_handles_query_error(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t1 (id INTEGER)"))

    def fake_read_sql(query, eng):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    report = db_utils.run_diagnostics(engine)
    assert report["connected"] is True
    assert report["errors"]


def test_run_diagnostics_handles_connection_error(monkeypatch):
    class FailingEngine:
        def connect(self):
            raise SQLAlchemyError("boom")

        dialect = type("d", (), {"name": "sqlite"})()

    report = db_utils.run_diagnostics(FailingEngine())
    assert report["connected"] is False
    assert report["errors"]
