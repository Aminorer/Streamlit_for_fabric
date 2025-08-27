import logging
import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from typing import Callable, Optional, List, Dict, Tuple, Set

load_dotenv("secret.env")

logger = logging.getLogger(__name__)

# Whitelist of SQL tables allowed for queries
ALLOWED_TABLES: Set[str] = set()


def _assert_allowed_table(table: str) -> None:
    """Raise ValueError if the given table name is not whitelisted."""
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not allowed.")


def _build_engine(server: str, database: str) -> Engine:
    """Create a SQLAlchemy engine for the given server and database."""

    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    driver = os.getenv("SQL_DRIVER")

    if driver is None:
        raise ValueError(
            "La variable d'environnement SQL_DRIVER est manquante."
        )
    driver = driver.replace(" ", "+")

    connection_string = (
        f"mssql+pyodbc://{user}:{password}@{server}:1433/{database}"
        f"?driver={driver}"
        "&authentication=ActiveDirectoryPassword"
        "&encrypt=yes"
        "&TrustServerCertificate=no"
    )
    return create_engine(connection_string)


def get_engine_hist() -> Engine:
    server = os.getenv("SQL_SERVER_HIST")
    database = os.getenv("SQL_DATABASE_HIST")
    if server is None:
        raise ValueError(
            "La variable d'environnement SQL_SERVER_HIST est manquante."
        )
    if database is None:
        raise ValueError(
            "La variable d'environnement SQL_DATABASE_HIST est manquante."
        )
    return _build_engine(server, database)


def get_engine_pred() -> Engine:
    server = os.getenv("SQL_SERVER_PRED")
    database = os.getenv("SQL_DATABASE_PRED")
    if server is None:
        raise ValueError(
            "La variable d'environnement SQL_SERVER_PRED est manquante."
        )
    if database is None:
        raise ValueError(
            "La variable d'environnement SQL_DATABASE_PRED est manquante."
        )
    return _build_engine(server, database)


def get_engine() -> Engine:
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    if server is None:
        raise ValueError(
            "La variable d'environnement SQL_SERVER est manquante."
        )
    if database is None:
        raise ValueError(
            "La variable d'environnement SQL_DATABASE est manquante."
        )
    return _build_engine(server, database)


@st.cache_data(show_spinner=False)
def find_hist_tables() -> List[str]:
    """Return historical table names matching the pattern fullsize_stock_hist_%."""
    engine = get_engine_hist()
    query = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME LIKE 'fullsize_stock_hist_%'"
    )
    try:
        df = pd.read_sql(query, engine)
        tables = df["TABLE_NAME"].tolist()
        return [t for t in tables if t in ALLOWED_TABLES]
    except SQLAlchemyError as e:
        logger.error(
            "Erreur lors de la récupération des tables historiques: %s", e
        )
        return []


@st.cache_data(show_spinner=False)
def find_pred_tables() -> List[str]:
    """Return prediction table names matching the pattern pred_%."""
    engine = get_engine_pred()
    query = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME LIKE 'pred_%'"
    )
    try:
        df = pd.read_sql(query, engine)
        tables = df["TABLE_NAME"].tolist()
        return [t for t in tables if t in ALLOWED_TABLES]
    except SQLAlchemyError as e:
        logger.error(
            "Erreur lors de la récupération des tables de prédiction: %s", e
        )
        return []


@st.cache_data(show_spinner=False)
def discover_platforms() -> Dict[str, List[str]]:
    """Discover available platforms and activities from table names.

    Table names are expected to follow the pattern ``prefix_platform_activity``
    where ``prefix`` is either ``fullsize_stock_hist`` or ``pred``.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of platform name to the list of activity types found for that
        platform.
    """

    tables = find_hist_tables() + find_pred_tables()
    platforms: Dict[str, set] = {}
    for tbl in tables:
        if tbl.startswith("fullsize_stock_hist_"):
            suffix = tbl.replace("fullsize_stock_hist_", "")
        elif tbl.startswith("pred_"):
            suffix = tbl.replace("pred_", "")
        else:
            continue

        parts = suffix.split("_")
        if len(parts) != 2:
            logger.warning("Table name '%s' does not match pattern '_XX_YY'", tbl)
            continue
        platform, activity = parts[0].lower(), parts[1].lower()
        platforms.setdefault(platform, set()).add(activity)

    return {plat: sorted(list(acts)) for plat, acts in sorted(platforms.items())}


@st.cache_data(show_spinner=False)
def get_matching_tables(platform: str, activity_type: str) -> Tuple[Optional[str], Optional[str]]:
    """Return historical and prediction table names for the given platform.

    Parameters
    ----------
    platform : str
        Platform identifier (e.g. ``amz``).
    activity_type : str
        Activity type identifier (e.g. ``man``, ``dis``, ``mixte``).

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        A tuple ``(historical_table, prediction_table)``. Each element may be
        ``None`` if no matching table is found.
    """

    suffix = f"{platform.lower()}_{activity_type.lower()}"
    hist_table = next(
        (tbl for tbl in find_hist_tables() if tbl.lower().endswith(suffix)), None
    )
    pred_table = next(
        (tbl for tbl in find_pred_tables() if tbl.lower().endswith(suffix)), None
    )
    return hist_table, pred_table


def validate_table_consistency(hist_table: str, pred_table: str) -> bool:
    """Validate that historical and prediction tables are consistent.

    The function checks that both tables refer to the same platform and
    activity type and that the schema of the historical table is a subset of
    the prediction table schema.

    Parameters
    ----------
    hist_table : str
        Name of the historical table (``fullsize_stock_hist_*``).
    pred_table : str
        Name of the prediction table (``pred_*``).

    Returns
    -------
    bool
        ``True`` if tables are consistent, ``False`` otherwise.
    """

    _assert_allowed_table(hist_table)
    _assert_allowed_table(pred_table)

    hist_suffix = hist_table.replace("fullsize_stock_hist_", "").lower()
    pred_suffix = pred_table.replace("pred_", "").lower()
    if hist_suffix != pred_suffix:
        logger.error(
            "Mismatch between historical table %s and prediction table %s",
            hist_table,
            pred_table,
        )
        return False

    try:
        hist_df = pd.read_sql(
            f"SELECT TOP 0 * FROM dbo.{hist_table}", get_engine_hist()
        )
        pred_df = pd.read_sql(
            f"SELECT TOP 0 * FROM dbo.{pred_table}", get_engine_pred()
        )
    except SQLAlchemyError as e:
        logger.error(
            "Erreur lors de la comparaison des tables %s et %s: %s",
            hist_table,
            pred_table,
            e,
        )
        return False

    hist_cols = set(hist_df.columns)
    pred_cols = set(pred_df.columns)
    if not hist_cols.issubset(pred_cols):
        logger.error(
            "Incompatible schemas for %s and %s: %s vs %s",
            hist_table,
            pred_table,
            sorted(hist_cols),
            sorted(pred_cols),
        )
        return False

    return True


@st.cache_data(show_spinner=False)
def load_hist_data(
    brands: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
):
    """Load historical data applying optional filters at the SQL level."""
    tables = find_hist_tables()
    if not tables:
        logger.error(
            "Aucune table historique trouvée correspondant au motif fullsize_stock_hist_%"
        )
        return pd.DataFrame()
    table_name = tables[0]
    _assert_allowed_table(table_name)
    engine = get_engine_hist()
    query = (
        "SELECT date_key, tyre_brand, tyre_season_french, tyre_fullsize, "
        f"Sum_stock_quantity, Avg_supplier_price_eur FROM dbo.{table_name} WHERE 1=1"
    )
    params: Dict[str, object] = {}
    if brands:
        placeholders = ",".join([f":brand{i}" for i in range(len(brands))])
        query += f" AND tyre_brand IN ({placeholders})"
        params.update({f"brand{i}": b for i, b in enumerate(brands)})
    if seasons:
        placeholders = ",".join([f":season{i}" for i in range(len(seasons))])
        query += f" AND tyre_season_french IN ({placeholders})"
        params.update({f"season{i}": s for i, s in enumerate(seasons)})
    if sizes:
        placeholders = ",".join([f":size{i}" for i in range(len(sizes))])
        query += f" AND tyre_fullsize IN ({placeholders})"
        params.update({f"size{i}": sz for i, sz in enumerate(sizes)})
    if start_date is not None:
        query += " AND date_key >= :start_date"
        params["start_date"] = start_date
    if end_date is not None:
        query += " AND date_key <= :end_date"
        params["end_date"] = end_date
    try:
        if params:
            df = pd.read_sql(query, engine, params=params)
        else:
            df = pd.read_sql(query, engine)
        df["date_key"] = pd.to_datetime(df["date_key"])
        return df
    except SQLAlchemyError as e:
        logger.error("Erreur lors du chargement des données historiques: %s", e)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_prediction_data(
    table_name: str,
    brands: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    sizes: Optional[List[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
):
    """Load prediction data applying optional filters at the SQL level.

    Parameters
    ----------
    table_name : str
        Name of the prediction table to query. Must be whitelisted in
        ``ALLOWED_TABLES``.
    brands, seasons, sizes : Optional[List[str]]
        Filters applied on corresponding columns.
    start_date, end_date : Optional[pd.Timestamp]
        Date range filters applied on ``date_key``.
    """

    _assert_allowed_table(table_name)
    engine = get_engine_pred()
    query = (
        "SELECT date_key, tyre_fullsize, tyre_brand, tyre_season_french, "
        "stock_prediction, price_prediction, ic_price_plus, ic_price_minus, "
        "ic_stock_plus, ic_stock_minus, prediction_confidence, stock_status, "
        "volatility_status, main_rupture_date, order_recommendation, "
        "tension_days, recommended_volume, optimal_order_date, "
        "last_safe_order_date, margin_opportunity_days, criticality_score, "
        "risk_level, stability_index, anomaly_alert, seasonal_factor, "
        "supply_chain_alert, volatility_type, procurement_urgency, "
        "price_jump_alert "
        f"FROM dbo.{table_name} WHERE 1=1"
    )
    params: Dict[str, object] = {}
    if brands:
        placeholders = ",".join([f":brand{i}" for i in range(len(brands))])
        query += f" AND tyre_brand IN ({placeholders})"
        params.update({f"brand{i}": b for i, b in enumerate(brands)})
    if seasons:
        placeholders = ",".join([f":season{i}" for i in range(len(seasons))])
        query += f" AND tyre_season_french IN ({placeholders})"
        params.update({f"season{i}": s for i, s in enumerate(seasons)})
    if sizes:
        placeholders = ",".join([f":size{i}" for i in range(len(sizes))])
        query += f" AND tyre_fullsize IN ({placeholders})"
        params.update({f"size{i}": sz for i, sz in enumerate(sizes)})
    if start_date is not None:
        query += " AND date_key >= :start_date"
        params["start_date"] = start_date
    if end_date is not None:
        query += " AND date_key <= :end_date"
        params["end_date"] = end_date

    try:
        if params:
            df = pd.read_sql(query, engine, params=params)
        else:
            df = pd.read_sql(query, engine)
        df["date_key"] = pd.to_datetime(df["date_key"])
        for col in ["main_rupture_date", "optimal_order_date", "last_safe_order_date"]:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        return df
    except SQLAlchemyError as e:
        logger.error("Erreur lors du chargement des données de prédiction: %s", e)
        return pd.DataFrame()


def prediction_table_exists(table_name: str) -> bool:
    """Check if a prediction table already exists in the database."""
    return table_name in find_pred_tables()


def save_dataframe_to_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Save a DataFrame to the specified SQL table."""
    _assert_allowed_table(table_name)
    engine = get_engine_pred()
    try:
        df.to_sql(
            table_name,
            con=engine,
            schema="dbo",
            index=False,
            if_exists="replace",
        )
        return df
    except SQLAlchemyError as e:
        logger.error(
            "Erreur lors de l'enregistrement de la table %s: %s", table_name, e
        )
        return pd.DataFrame()


def generate_codex_predictions(
    df_hist: pd.DataFrame,
    horizon: int = 30,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Generate stock and price predictions using a RandomForest model.

    Parameters
    ----------
    df_hist : pd.DataFrame
        Historical data used for training.
    horizon : int, optional
        Number of days to predict, by default 30.
    progress_callback : Callable[[float], None], optional
        Function called with progress in [0, 1] after each iteration.
    """
    df = df_hist.dropna(subset=["Sum_stock_quantity", "Avg_supplier_price_eur"]).copy()
    df["date_ord"] = df["date_key"].map(pd.Timestamp.toordinal)

    cat_cols = ["tyre_brand", "tyre_season_french", "tyre_fullsize"]
    num_cols = ["date_ord"]

    preproc = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    stock_model = Pipeline(
        [
            ("preproc", preproc),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    price_model = Pipeline(
        [
            ("preproc", preproc),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    X = df[cat_cols + num_cols]
    stock_model.fit(X, df["Sum_stock_quantity"])
    price_model.fit(X, df["Avg_supplier_price_eur"])

    max_date = df["date_key"].max()
    future_dates = pd.date_range(max_date + pd.Timedelta(days=1), periods=horizon)

    combos = df[cat_cols].drop_duplicates()

    # Vectorized generation of future features instead of looping per day
    future_df = (
        combos.assign(key=1)
        .merge(pd.DataFrame({"date_key": future_dates, "key": 1}), on="key")
        .drop("key", axis=1)
    )
    future_df["date_ord"] = future_df["date_key"].map(pd.Timestamp.toordinal)

    stock_pred = stock_model.predict(future_df[cat_cols + num_cols])
    price_pred = price_model.predict(future_df[cat_cols + num_cols])

    future_df["stock_prediction"] = stock_pred
    future_df["price_prediction"] = price_pred

    if progress_callback is not None:
        progress_callback(1.0)

    return future_df[["date_key"] + cat_cols + ["stock_prediction", "price_prediction"]]
