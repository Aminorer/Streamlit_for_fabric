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


def get_engine():
    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
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


def load_hist_data():
    engine = get_engine()
    query = (
        "SELECT date_key, tyre_brand, tyre_season_french, tyre_fullsize, "
        "Sum_stock_quantity, Avg_supplier_price_eur FROM dbo.fullsize_stock_hist"
    )
    df = pd.read_sql(query, engine)
    df["date_key"] = pd.to_datetime(df["date_key"])
    return df


def prediction_table_exists(table_name: str) -> bool:
    """Check if a prediction table already exists in the database."""
    engine = get_engine()
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
    engine = get_engine()
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
    all_preds = []
    for idx, d in enumerate(future_dates, start=1):
        features = combos.copy()
        features["date_ord"] = d.toordinal()
        stock_pred = stock_model.predict(features[cat_cols + num_cols])
        price_pred = price_model.predict(features[cat_cols + num_cols])
        out = features.assign(
            date_key=d,
            stock_prediction=stock_pred,
            price_prediction=price_pred,
        )[["date_key"] + cat_cols + ["stock_prediction", "price_prediction"]]
        all_preds.append(out)
        if progress_callback is not None:
            progress_callback(idx / len(future_dates))

    return pd.concat(all_preds, ignore_index=True)
