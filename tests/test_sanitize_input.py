import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from input_utils import sanitize_input, sanitize_list


def test_sanitize_input_valid():
    assert sanitize_input("Marque-1") == "Marque-1"


@pytest.mark.parametrize("value", ["", "bad;drop", "abc$", "a" * 101])
def test_sanitize_input_invalid(value):
    with pytest.raises(ValueError):
        sanitize_input(value)


def test_sanitize_list_invalid():
    with pytest.raises(ValueError):
        sanitize_list(["ok", "not@ok"])
