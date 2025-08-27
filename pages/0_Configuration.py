import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from db_utils import get_engine_hist, get_engine_pred

from input_utils import sanitize_input

PLATFORM_PATTERN = r"_([A-Z]{2}_[0-9]{2})"


def _scan_tables(get_engine) -> pd.DataFrame:
    """Return dataframe of table names and platforms for a given engine function."""
    engine = get_engine()
    query = "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema='dbo'"
    try:
        df = pd.read_sql(query, engine)
    except SQLAlchemyError:
        return pd.DataFrame(columns=["table_name", "platform"])
    df["platform"] = df["table_name"].str.extract(PLATFORM_PATTERN)
    return df


@st.cache_data
def list_hist_tables() -> pd.DataFrame:
    return _scan_tables(get_engine_hist)


@st.cache_data
def list_pred_tables() -> pd.DataFrame:
    return _scan_tables(get_engine_pred)


def extract_platform(table_name: str) -> str | None:
    match = re.search(PLATFORM_PATTERN, table_name)
    return match.group(1) if match else None


def fetch_distinct_values(table: str, column: str) -> List:
    engine = get_engine_hist()
    stmt = text(f"SELECT DISTINCT {column} FROM dbo.{table}")
    try:
        df = pd.read_sql(stmt, engine)
    except SQLAlchemyError:
        return []
    return df[column].dropna().tolist()


def fetch_date_bounds(table: str, column: str) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    engine = get_engine_hist()
    stmt = text(f"SELECT MIN({column}) AS min_date, MAX({column}) AS max_date FROM dbo.{table}")
    try:
        df = pd.read_sql(stmt, engine)
    except SQLAlchemyError:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df["min_date"]).iloc[0], pd.to_datetime(df["max_date"]).iloc[0]


def load_filter_values(table: str) -> Dict[str, Tuple[str, List]]:
    engine = get_engine_hist()
    try:
        columns = pd.read_sql(f"SELECT TOP 0 * FROM dbo.{table}", engine).columns
    except SQLAlchemyError:
        return {}
    mapping = {
        "Activité": ["activity", "tyre_activity"],
        "Marque": ["tyre_brand", "brand"],
        "Saison": ["tyre_season_french", "season"],
        "Taille": ["tyre_fullsize", "size"],
        "Dates": ["date_key", "date"],
    }
    filters = {}
    for label, candidates in mapping.items():
        for col in candidates:
            if col in columns:
                if label == "Dates":
                    bounds = fetch_date_bounds(table, col)
                    if bounds:
                        filters[label] = (col, bounds)
                else:
                    values = fetch_distinct_values(table, col)
                    if values:
                        filters[label] = (col, values)
                break
    return filters


def main():
    st.title("Configuration")

    hist_df = list_hist_tables()
    pred_df = list_pred_tables()

    platforms = sorted(
        set(hist_df["platform"].dropna()).union(set(pred_df["platform"].dropna()))
    )
    platform = (
        sanitize_input(st.selectbox("Plateforme", platforms)) if platforms else None
    )

    hist_tables = (
        hist_df[hist_df["platform"] == platform]["table_name"].tolist()
        if platform
        else hist_df["table_name"].tolist()
    )
    pred_tables = (
        pred_df[pred_df["platform"] == platform]["table_name"].tolist()
        if platform
        else pred_df["table_name"].tolist()
    )

    hist_table = (
        sanitize_input(st.selectbox("Table historique", hist_tables))
        if hist_tables
        else None
    )
    pred_table = (
        sanitize_input(st.selectbox("Table de prédiction", pred_tables))
        if pred_tables
        else None
    )

    if hist_table and pred_table:
        plat_hist = extract_platform(hist_table)
        plat_pred = extract_platform(pred_table)
        if plat_hist == plat_pred:
            st.success(f"Tables correspondantes pour la plateforme {plat_hist}")
        else:
            st.error("Les tables sélectionnées n'appartiennent pas à la même plateforme")

    if hist_table:
        filters = load_filter_values(hist_table)
        if "Activité" in filters:
            st.multiselect("Activité", filters["Activité"][1])
        if "Dates" in filters:
            min_d, max_d = filters["Dates"][1]
            st.date_input("Dates", (min_d.date(), max_d.date()))
        if "Marque" in filters:
            st.multiselect("Marque", filters["Marque"][1])
        if "Saison" in filters:
            st.multiselect("Saison", filters["Saison"][1])
        if "Taille" in filters:
            st.multiselect("Taille", filters["Taille"][1])


if __name__ == "__main__":
    main()
