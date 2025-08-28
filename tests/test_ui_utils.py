import sys
from pathlib import Path
import types

import streamlit as st
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils
from ui_utils import (
    setup_sidebar_filters,
    setup_prediction_comparison_filters,
    hex_to_rgb,
)


def test_setup_sidebar_filters_discovers_platforms(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "discover_platforms",
        lambda: {"amz": ["man"], "ebay": ["dis"]},
    )

    captured = {}

    def selectbox(label, options):
        captured["platform_options"] = options
        return options[0]

    dummy_sidebar = types.SimpleNamespace(
        selectbox=selectbox,
        radio=lambda label, options: options[0],
        date_input=lambda label, value=None: value,
        multiselect=lambda label, options, default=None: default or [],
    )
    monkeypatch.setattr(st, "sidebar", dummy_sidebar)

    filters = setup_sidebar_filters()
    assert captured["platform_options"] == ["amz", "ebay"]
    assert filters["platform"] == "amz"


def test_setup_prediction_comparison_filters(monkeypatch):
    df = pd.DataFrame(
        {
            "tyre_brand": ["A"],
            "tyre_season_french": ["Été"],
            "tyre_fullsize": ["195"],
        }
    )

    dummy_sidebar = types.SimpleNamespace(
        date_input=lambda label, value=None: value,
        multiselect=lambda label, options, default=None: default or [],
    )
    monkeypatch.setattr(st, "sidebar", dummy_sidebar)

    filters = setup_prediction_comparison_filters(df)
    assert filters["brands"] == ["A"]
    assert filters["seasons"] == ["Été"]
    assert filters["sizes"] == ["195"]


def test_hex_to_rgb_valid():
    assert hex_to_rgb("#ffffff") == (255, 255, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)


def test_hex_to_rgb_invalid():
    with pytest.raises(ValueError):
        hex_to_rgb("zzz")
