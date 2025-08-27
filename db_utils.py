import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from typing import Callable, Optional

load_dotenv("secret.env")


def _build_engine(database: str) -> Engine:
    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    server = os.getenv("SQL_SERVER")
    driver = os.getenv("SQL_DRIVER")

    if driver is None:
        raise ValueError("La variable d'environnement SQL_DRIVER est manquante.")
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
    database = os.getenv("SQL_DATABASE_HIST")
    if database is None:
        raise ValueError("La variable d'environnement SQL_DATABASE_HIST est manquante.")
    return _build_engine(database)


def get_engine_pred() -> Engine:
    database = os.getenv("SQL_DATABASE_PRED")
    if database is None:
        raise ValueError("La variable d'environnement SQL_DATABASE_PRED est manquante.")
    return _build_engine(database)


def get_engine() -> Engine:
    database = os.getenv("SQL_DATABASE")
    if database is None:
        raise ValueError("La variable d'environnement SQL_DATABASE est manquante.")
    return _build_engine(database)


def load_hist_data():
    engine = get_engine_hist()
    query = (
        "SELECT date_key, tyre_brand, tyre_season_french, tyre_fullsize, "
        "Sum_stock_quantity, Avg_supplier_price_eur FROM dbo.fullsize_stock_hist"
    )
    df = pd.read_sql(query, engine)
    df["date_key"] = pd.to_datetime(df["date_key"])
    return df


def prediction_table_exists(table_name: str) -> bool:
    """Check if a prediction table already exists in the database."""
    engine = get_engine_pred()
    query = (
        "SELECT 1 FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?"
    )
    try:
        df = pd.read_sql(query, engine, params=[table_name])
    except SQLAlchemyError:
        return False
    return not df.empty


def save_dataframe_to_table(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the specified SQL table."""
    engine = get_engine_pred()
    df.to_sql(table_name, con=engine, schema="dbo", index=False, if_exists="replace")


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
