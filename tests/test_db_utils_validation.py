import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils


def test_validate_table_name_valid(monkeypatch):
    """Valid table names pass regex and existence checks."""

    class DummyInspector:
        def has_table(self, table, schema):
            return True

    monkeypatch.setattr(db_utils, "inspect", lambda engine: DummyInspector())
    db_utils.validate_table_name("valid_table_123", engine=object())


@pytest.mark.parametrize("name", ["bad;name", "bad name", "name$", ""]) 
def test_validate_table_name_invalid(name):
    """Invalid table names raise ``ValueError`` due to regex check."""

    with pytest.raises(ValueError):
        db_utils.validate_table_name(name)


def test_validate_table_name_missing_table(monkeypatch):
    """Valid name but missing table raises ``ValueError`` when engine is provided."""

    class DummyInspector:
        def has_table(self, table, schema):
            return False

    monkeypatch.setattr(db_utils, "inspect", lambda engine: DummyInspector())
    with pytest.raises(ValueError):
        db_utils.validate_table_name("missing_table", engine=object())


def test_save_dataframe_to_table_invalid_name():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        db_utils.save_dataframe_to_table(df, "bad name")
