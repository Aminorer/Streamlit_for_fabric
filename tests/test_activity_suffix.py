from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from db_utils import get_activity_suffix


@pytest.mark.parametrize(
    "value,expected",
    [
        ("man", "man"),
        ("Manufacturing", "man"),
        ("dis", "dis"),
        ("Distribution", "dis"),
        ("mixte", "mixte"),
        ("Mixed", "mixte"),
    ],
)
def test_get_activity_suffix_valid(value, expected):
    assert get_activity_suffix(value) == expected


@pytest.mark.parametrize("invalid", ["", "unknown", "abc"])
def test_get_activity_suffix_invalid(invalid):
    with pytest.raises(ValueError):
        get_activity_suffix(invalid)
