import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import db_utils
from input_utils import sanitize_input, sanitize_list


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert HEX color (e.g. ``"#ff00aa"``) to an RGB tuple."""
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError("Invalid HEX color")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return r, g, b


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
    platforms = sorted(db_utils.discover_platforms().keys())
    platform = (
        st.sidebar.selectbox("Plateforme", platforms)
        if platforms
        else st.sidebar.text_input("Plateforme", "")
    )
    platform = sanitize_input(platform) if platform else ""

    activity_type = st.sidebar.radio("Type d'activité", ["Historique", "Prédiction"])
    activity_type = sanitize_input(activity_type)
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

    # Sanitize list selections
    brands = sanitize_list(brands)
    seasons = sanitize_list(seasons)
    sizes = sanitize_list(sizes)

    return {
        "platform": platform,
        "activity_type": activity_type,
        "date": date,
        "brands": brands,
        "seasons": seasons,
        "sizes": sizes,
    }


def setup_prediction_comparison_filters(
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Render sidebar filters for the comparative analysis page.

    Parameters
    ----------
    df : Optional[pandas.DataFrame]
        DataFrame used to populate brand, season and size options if
        provided. Only the presence of relevant columns is checked.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the selected filter values including a
        date range.
    """

    default_start = datetime.date.today() - datetime.timedelta(days=30)
    default_end = datetime.date.today()
    start_date, end_date = st.sidebar.date_input(
        "Période", (default_start, default_end)
    )

    brand_options: List[str] = []
    season_options: List[str] = []
    size_options: List[str] = []

    if df is not None:
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

    brands = sanitize_list(brands)
    seasons = sanitize_list(seasons)
    sizes = sanitize_list(sizes)

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    return {
        "start_date": start_ts,
        "end_date": end_ts,
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
