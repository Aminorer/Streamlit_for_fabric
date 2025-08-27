import sys
from pathlib import Path
import logging
import pytest
from sqlalchemy.exc import SQLAlchemyError

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_build_engine_attempt_sequence(monkeypatch, caplog):
    attempts = []

    def fake_create_engine(conn_str, **kwargs):
        attempt_index = len(attempts)
        attempts.append(conn_str)

        class DummyConn:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

        class DummyEngine:
            def connect(self):
                if attempt_index < 2:
                    raise SQLAlchemyError("boom")
                return DummyConn()

        return DummyEngine()

    monkeypatch.setattr(db_utils, "create_engine", fake_create_engine)
    monkeypatch.setenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("SQL_USER", "u")
    monkeypatch.setenv("SQL_PASSWORD", "p")

    with caplog.at_level(logging.INFO):
        engine = db_utils._build_engine("srv", "db")

    assert engine is not None
    assert len(attempts) == 3
    assert "authentication=ActiveDirectoryInteractive" in attempts[1]
    assert "authentication=ActiveDirectoryIntegrated" in attempts[2]
    assert all("authentication=ActiveDirectoryPassword" not in a for a in attempts)
    assert "SQL connection using ActiveDirectoryIntegrated succeeded" in caplog.text
