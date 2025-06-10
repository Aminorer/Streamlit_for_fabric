import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

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
