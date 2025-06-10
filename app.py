import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

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
    df = pd.read_sql("SELECT date_key, Sum_stock_quantity, Avg_supplier_price_eur FROM dbo.fullsize_stock_hist", engine)
    df['date_key'] = pd.to_datetime(df['date_key'])
    return df

def plot_stock_sum(df):
    daily_stock = df.groupby("date_key")["Sum_stock_quantity"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stock["date_key"],
        y=daily_stock["Sum_stock_quantity"],
        mode="lines+markers",
        name="Stock total",
    ))
    fig.update_layout(title="Évolution du stock total", xaxis_title="Date", yaxis_title="Stock", height=400)
    return fig

def plot_price_avg(df):
    daily_price = df.groupby("date_key")["Avg_supplier_price_eur"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_price["date_key"],
        y=daily_price["Avg_supplier_price_eur"],
        mode="lines+markers",
        name="Prix moyen fournisseur",
    ))
    fig.update_layout(title="Évolution du prix moyen fournisseur", xaxis_title="Date", yaxis_title="Prix (€)", height=400)
    return fig

def main():
    st.set_page_config(page_title="Historique des stocks", layout="wide")
    st.title("Historique des stocks")

    df = load_hist_data()

    if df.empty:
        st.error("Aucune donnée disponible.")
        return

    
    fig_stock = plot_stock_sum(df)
    st.plotly_chart(fig_stock, use_container_width=True)

    
    fig_price = plot_price_avg(df)
    st.plotly_chart(fig_price, use_container_width=True)

if __name__ == "__main__":
    main()
