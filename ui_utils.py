import datetime
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import db_utils


def _discover_platforms() -> List[str]:
    """Discover platform names from available database tables.

    Returns
    -------
    List[str]
        Sorted unique platform names extracted from historical and
        prediction table names.
    """
    tables = db_utils.find_hist_tables() + db_utils.find_pred_tables()
    platforms = {
        tbl.replace("fullsize_stock_hist_", "").replace("pred_", "")
        for tbl in tables
    }
    return sorted(platforms)


def setup_sidebar_filters(df: Optional[object] = None) -> Dict[str, Any]:
    """Render common sidebar filters and return selected values.

    Parameters
    ----------
    df : Optional[pandas.DataFrame]
        DataFrame used to populate brand, season and size options if
        provided. The function only checks for the relevant column names
        and ignores the DataFrame structure otherwise.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the selected filter values.
    """
    platforms = _discover_platforms()
    platform = (
        st.sidebar.selectbox("Plateforme", platforms)
        if platforms
        else st.sidebar.text_input("Plateforme", "")
    )

    activity_type = st.sidebar.radio("Type d'activité", ["Historique", "Prédiction"])
    date = st.sidebar.date_input("Date", datetime.date.today())

    brand_options: List[str] = []
    season_options: List[str] = []
    size_options: List[str] = []

    if df is not None:
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - pandas always available in app
            pd = None  # type: ignore
        if pd is not None:
            if "tyre_brand" in df:
                brand_options = sorted(df["tyre_brand"].dropna().unique().tolist())
            if "tyre_season_french" in df:
                season_options = sorted(
                    df["tyre_season_french"].dropna().unique().tolist()
                )
            if "tyre_fullsize" in df:
                size_options = sorted(df["tyre_fullsize"].dropna().unique().tolist())

    brands = st.sidebar.multiselect("Marques", brand_options, default=brand_options)
    seasons = st.sidebar.multiselect("Saisons", season_options, default=season_options)
    sizes = st.sidebar.multiselect("Tailles", size_options, default=size_options)

    return {
        "platform": platform,
        "activity_type": activity_type,
        "date": date,
        "brands": brands,
        "seasons": seasons,
        "sizes": sizes,
    }


def display_dataframe(df: pd.DataFrame, rows_per_page: int = 1000) -> None:
    """Display a DataFrame with pagination when it exceeds 10k rows.

    A progress bar indicates the portion of data currently visible.
    """
    total_rows = len(df)
    if total_rows > 10_000:
        total_pages = math.ceil(total_rows / rows_per_page)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start = (page - 1) * rows_per_page
        end = start + rows_per_page
        progress = st.progress(0)
        progress.progress(min(int(end / total_rows * 100), 100))
        st.dataframe(df.iloc[start:end], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
