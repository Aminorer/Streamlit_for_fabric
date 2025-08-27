import sys
from pathlib import Path
import importlib
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _import_db_utils(monkeypatch, allowed: str, expected: str):
    monkeypatch.setenv("ALLOWED_TABLES", allowed)
    monkeypatch.setenv("EXPECTED_TABLES", expected)
    if "db_utils" in sys.modules:
        del sys.modules["db_utils"]
    return importlib.import_module("db_utils")


def test_missing_expected_tables_raises(monkeypatch):
    with pytest.raises(ValueError):
        _import_db_utils(monkeypatch, "tbl1", "tbl1,tbl2")


def test_expected_tables_ok(monkeypatch):
    _import_db_utils(monkeypatch, "tbl1,tbl2", "tbl1")
