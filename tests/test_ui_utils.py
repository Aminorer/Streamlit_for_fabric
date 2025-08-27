import sys
from pathlib import Path
import types

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
import db_utils
from ui_utils import setup_sidebar_filters


def test_setup_sidebar_filters_discovers_platforms(monkeypatch):
    monkeypatch.setattr(
        db_utils,
        "find_hist_tables",
        lambda: ["fullsize_stock_hist_amz", "fullsize_stock_hist_ebay"],
    )
    monkeypatch.setattr(db_utils, "find_pred_tables", lambda: ["pred_amz"])

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
