import logging
import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, NoInspectionAvailable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from typing import Any, Callable, Optional, List, Dict, Tuple

load_dotenv("secret.env")

logger = logging.getLogger(__name__)


_TABLE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_table_name(table: str, engine: Optional[Engine] = None) -> None:
    """Validate a table name and optionally ensure it exists.

    Parameters
    ----------
    table : str
        Name of the table to validate.
    engine : Optional[Engine]
        SQLAlchemy engine used to verify the table exists. If ``None`` the
        existence check is skipped.
    """

    if not _TABLE_NAME_RE.fullmatch(table):
        raise ValueError(f"Invalid table name '{table}'.")

    if engine is None:
        return

    try:
        inspector = inspect(engine)
        if not inspector.has_table(table, schema="dbo"):
            raise ValueError(f"Table '{table}' does not exist.")
    except NoInspectionAvailable:
        # Engine could be a mock object in tests; skip existence check.
        return
    except SQLAlchemyError as e:
        logger.exception(
            "Erreur lors de l'inspection de la table %s", table
        )
        raise ValueError(f"Table '{table}' does not exist.") from e


def _build_engine(server: str, database: str) -> Engine:
    """Create a SQLAlchemy engine for the given server and database.

    Multiple authentication methods are attempted in a specific order. The
    first successful connection is returned.
    """

    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    driver = os.getenv("SQL_DRIVER")

    if driver is None:
        raise ValueError(
            "La variable d'environnement SQL_DRIVER est manquante."
        )
    driver = driver.replace(" ", "+")

    base_connection_string = (
        f"mssql+pyodbc://{user}:{password}@{server}:1433/{database}"
        f"?driver={driver}"
        "&encrypt=yes"
        "&TrustServerCertificate=no"
        "&Connect Timeout=30"
    )

    auth_methods = [
        ("", "no authentication parameter"),
        ("&authentication=ActiveDirectoryInteractive", "ActiveDirectoryInteractive"),
        ("&authentication=ActiveDirectoryIntegrated", "ActiveDirectoryIntegrated"),
        ("&authentication=ActiveDirectoryPassword", "ActiveDirectoryPassword"),
    ]

    engine_kwargs = {
        "connect_args": {"timeout": 30},
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
    }

    for suffix, label in auth_methods:
        connection_string = base_connection_string + suffix
        logger.info("Attempting SQL connection using %s", label)
        try:
            engine = create_engine(connection_string, **engine_kwargs)
            with engine.connect() as conn:
                pass
            logger.info("SQL connection using %s succeeded", label)
            return engine
        except SQLAlchemyError:
            logger.exception("SQL connection using %s failed", label)

    raise SQLAlchemyError("All SQL connection attempts failed.")


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


def _classify_tables(engine, marker_column: str) -> List[str]:
    """Return table names containing ``marker_column``.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the target database.
    marker_column:
        Column name whose presence identifies the table type.
    """

    table_query = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'dbo'"
    )
    df_tables = pd.read_sql(table_query, engine)
    tables = df_tables["TABLE_NAME"].tolist()
    matched: List[str] = []
    for tbl in tables:
        col_query = (
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = :table"
        )
        df_cols = pd.read_sql(text(col_query), engine, params={"table": tbl})
        cols = {c.lower() for c in df_cols["COLUMN_NAME"]}
        if marker_column.lower() in cols:
            matched.append(tbl)
    return matched


def find_hist_tables() -> List[str]:
    """Return historical table names identified by ``Sum_stock_quantity`` column."""
    engine = get_engine_hist()
    try:
        return _classify_tables(engine, "Sum_stock_quantity")
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors de la récupération des tables historiques"
        )
        raise


def find_pred_tables() -> List[str]:
    """Return prediction table names identified by ``stock_prediction`` column."""
    engine = get_engine_pred()
    try:
        return _classify_tables(engine, "stock_prediction")
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors de la récupération des tables de prédiction"
        )
        raise


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
def discover_prediction_weeks(platform: str) -> List[str]:
    """Return available week labels for prediction tables of a platform.

    Table names are expected to follow the pattern
    ``pred_<platform>_<activity>_<YYYYMMDD>`` where the date corresponds to
    the start of the prediction week. The returned labels are formatted as
    ``DD/MM/YYYY - Semaine X`` where ``X`` is the ISO week number.

    Parameters
    ----------
    platform : str
        Platform identifier (e.g. ``amz``).

    Returns
    -------
    List[str]
        Sorted list of unique week labels available for the platform.
    """

    tables = find_pred_tables()
    prefix = f"pred_{platform.lower()}_"
    weeks: Dict[pd.Timestamp, str] = {}
    for tbl in tables:
        if not tbl.lower().startswith(prefix):
            continue
        parts = tbl.split("_")
        if len(parts) < 4:
            continue
        date_part = parts[-1]
        try:
            date = pd.to_datetime(date_part, format="%Y%m%d", errors="raise")
        except ValueError:
            continue
        week_label = f"{date.strftime('%d/%m/%Y')} - Semaine {date.isocalendar().week}"
        weeks[date] = week_label

    return [weeks[d] for d in sorted(weeks)]


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

    validate_table_name(hist_table)
    validate_table_name(pred_table)
    engine_hist = get_engine_hist()
    engine_pred = get_engine_pred()
    validate_table_name(hist_table, engine_hist)
    validate_table_name(pred_table, engine_pred)

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
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors de la comparaison des tables %s et %s",
            hist_table,
            pred_table,
        )
        raise

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


def _compose_select_clause(
    engine: Engine, table_name: str, column_map: Dict[str, List[str]]
) -> Tuple[str, List[str]]:
    """Return a SELECT clause including only existing columns.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine used to inspect the table.
    table_name : str
        Table to inspect.
    column_map : Dict[str, List[str]]
        Mapping of canonical column names to optional aliases. The first
        matching alias found in the table will be selected and aliased to the
        canonical name if necessary.

    Returns
    -------
    Tuple[str, List[str]]
        A tuple ``(select_clause, selected_columns)`` where ``select_clause``
        is the comma-separated list to include in a ``SELECT`` statement and
        ``selected_columns`` is the list of canonical columns actually
        selected.
    """

    try:
        inspector = inspect(engine)
        cols_info = inspector.get_columns(table_name, schema="dbo")
        existing = {c["name"].lower(): c["name"] for c in cols_info}
    except NoInspectionAvailable as e:
        logger.warning("Could not inspect columns for %s: %s", table_name, e)
        existing = {col.lower(): col for col in column_map}
    except SQLAlchemyError:
        logger.exception("Could not inspect columns for %s", table_name)
        existing = {col.lower(): col for col in column_map}

    select_parts: List[str] = []
    selected: List[str] = []
    for canonical, aliases in column_map.items():
        candidates = [canonical] + [a for a in aliases]
        candidates = [c.lower() for c in candidates]
        found = None
        for cand in candidates:
            if cand in existing:
                found = existing[cand]
                break
        if found is not None:
            if found != canonical:
                select_parts.append(f"{found} AS {canonical}")
            else:
                select_parts.append(canonical)
            selected.append(canonical)

    return ", ".join(select_parts), selected


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
    validate_table_name(table_name)
    engine = get_engine_hist()
    validate_table_name(table_name, engine)

    col_map = {
        "date_key": [],
        "tyre_brand": [],
        "tyre_season_french": [],
        "tyre_fullsize": [],
        "Sum_stock_quantity": ["sum_stock_quantity"],
        "Avg_supplier_price_eur": ["avg_supplier_price_eur"],
    }
    select_clause, _ = _compose_select_clause(engine, table_name, col_map)
    query = f"SELECT {select_clause} FROM dbo.{table_name} WHERE 1=1"
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
        if "date_key" in df:
            df["date_key"] = pd.to_datetime(df["date_key"])
        return df
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors du chargement des données historiques"
        )
        raise


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
        Name of the prediction table to query.
    brands, seasons, sizes : Optional[List[str]]
        Filters applied on corresponding columns.
    start_date, end_date : Optional[pd.Timestamp]
        Date range filters applied on ``date_key``.
    """

    validate_table_name(table_name)
    if not prediction_table_exists(table_name):
        raise ValueError(f"Table '{table_name}' does not exist.")
    engine = get_engine_pred()
    validate_table_name(table_name, engine)
    col_map = {
        "date_key": [],
        "tyre_fullsize": [],
        "tyre_brand": [],
        "tyre_season_french": [],
        "stock_prediction": [],
        "price_prediction": [],
        "ic_price_plus": [],
        "ic_price_minus": [],
        "ic_stock_plus": [],
        "ic_stock_minus": [],
        "prediction_confidence": [],
        "stock_status": [],
        "volatility_status": [],
        "main_rupture_date": [],
        "order_recommendation": [],
        "tension_days": [],
        "recommended_volume": [],
        "optimal_order_date": [],
        "last_safe_order_date": [],
        "margin_opportunity_days": [],
        "criticality_score": [],
        "risk_level": [],
        "stability_index": [],
        "anomaly_alert": [],
        "seasonal_factor": [],
        "supply_chain_alert": [],
        "volatility_type": [],
        "procurement_urgency": [],
        "price_jump_alert": [],
    }
    select_clause, _ = _compose_select_clause(engine, table_name, col_map)
    query = f"SELECT {select_clause} FROM dbo.{table_name} WHERE 1=1"
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
        if "date_key" in df:
            df["date_key"] = pd.to_datetime(df["date_key"])
        for col in ["main_rupture_date", "optimal_order_date", "last_safe_order_date"]:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        return df
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors du chargement des données de prédiction"
        )
        raise


def prediction_table_exists(table_name: str) -> bool:
    """Check if a prediction table already exists in the database."""
    return table_name in find_pred_tables()


def load_multi_week_predictions(
    platform: str,
    activity_type: str,
    selected_weeks: List[str],
    filters: Dict[str, Optional[Any]],
) -> Dict[str, pd.DataFrame]:
    """Load prediction data for multiple weeks.

    Parameters
    ----------
    platform : str
        Platform identifier (e.g. ``amz``).
    activity_type : str
        Activity type identifier (e.g. ``man``).
    selected_weeks : List[str]
        Week labels as returned by :func:`discover_prediction_weeks`.
    filters : Dict[str, Optional[Any]]
        Mapping containing optional filter lists for ``brands``, ``seasons`` and
        ``sizes`` as well as ``start_date`` and ``end_date`` timestamps.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of week label to the loaded DataFrame.
    """

    tables = find_pred_tables()
    prefix = f"pred_{platform.lower()}_{activity_type.lower()}_"
    week_to_table: Dict[str, str] = {}
    for tbl in tables:
        if not tbl.lower().startswith(prefix):
            continue
        parts = tbl.split("_")
        if len(parts) < 4:
            continue
        date_part = parts[-1]
        try:
            date = pd.to_datetime(date_part, format="%Y%m%d", errors="raise")
        except ValueError:
            continue
        label = f"{date.strftime('%d/%m/%Y')} - Semaine {date.isocalendar().week}"
        week_to_table[label] = tbl

    result: Dict[str, pd.DataFrame] = {}
    for week in selected_weeks:
        table = week_to_table.get(week)
        if table is None:
            continue
        df = load_prediction_data(
            table,
            brands=filters.get("brands"),
            seasons=filters.get("seasons"),
            sizes=filters.get("sizes"),
            start_date=filters.get("start_date"),
            end_date=filters.get("end_date"),
        )
        result[week] = df
    return result


def get_table_columns(table_name: str, engine: Optional[Engine] = None) -> List[str]:
    """Return the list of columns for the given table.

    Parameters
    ----------
    table_name : str
        Name of the table to inspect.
    engine : Optional[Engine]
        SQLAlchemy engine used for inspection. If ``None`` the prediction
        database engine is used.

    Returns
    -------
    List[str]
        Names of columns present in the table.
    """

    validate_table_name(table_name)
    engine = engine or get_engine_pred()
    try:
        inspector = inspect(engine)
        cols_info = inspector.get_columns(table_name, schema="dbo")
        return [c["name"] for c in cols_info]
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors de l'inspection de la table %s", table_name
        )
        raise


def run_diagnostics(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Run connectivity and data checks on the target database.

    The function attempts to connect, lists available tables with their
    columns, and performs a small sample query on each table.

    Parameters
    ----------
    engine : Optional[Engine]
        Engine instance to use. If ``None`` a new engine is created via
        :func:`get_engine`.

    Returns
    -------
    Dict[str, Any]
        Structured report containing connectivity status, table metadata,
        sample rows, and any errors encountered.
    """

    report: Dict[str, Any] = {
        "connected": False,
        "tables": {},
        "samples": {},
        "errors": [],
    }

    try:
        engine = engine or get_engine()
        with engine.connect():
            report["connected"] = True

        inspector = inspect(engine)
        schema = "dbo" if engine.dialect.name == "mssql" else None
        tables = inspector.get_table_names(schema=schema)
        for tbl in tables:
            try:
                cols = [
                    col["name"] for col in inspector.get_columns(tbl, schema=schema)
                ]
                report["tables"][tbl] = cols
                if engine.dialect.name == "mssql":
                    query = f"SELECT TOP 5 * FROM {schema}.{tbl}" if schema else f"SELECT TOP 5 * FROM {tbl}"
                else:
                    query = f"SELECT * FROM {tbl} LIMIT 5"
                report["samples"][tbl] = pd.read_sql(query, engine)
            except SQLAlchemyError:
                logger.exception("Sample query failed for table %s", tbl)
                report["errors"].append(f"table {tbl}")
    except SQLAlchemyError as e:
        logger.exception("Diagnostics failed")
        report["errors"].append(str(e))

    return report


def save_dataframe_to_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Save a DataFrame to the specified SQL table."""
    validate_table_name(table_name)
    engine = get_engine_pred()
    validate_table_name(table_name, engine)
    try:
        df.to_sql(
            table_name,
            con=engine,
            schema="dbo",
            index=False,
            if_exists="replace",
        )
        return df
    except SQLAlchemyError:
        logger.exception(
            "Erreur lors de l'enregistrement de la table %s", table_name
        )
        raise


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
