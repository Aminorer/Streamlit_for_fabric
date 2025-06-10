import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db_utils import load_hist_data, get_engine


def list_prediction_tables():
    engine = get_engine()
    query = (
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES "
        "WHERE table_schema = 'dbo' AND table_name LIKE 'fullsize_stock_pred%'"
    )
    df = pd.read_sql(query, engine)
    return df["table_name"].tolist()


def load_prediction_data(table_name):
    engine = get_engine()
    query = (
        f"SELECT date_key, stock_prediction, price_prediction "
        f"FROM dbo.{table_name}"
    )
    df = pd.read_sql(query, engine)
    df["date_key"] = pd.to_datetime(df["date_key"])
    return df


def aggregate_predictions(df):
    agg = df.groupby("date_key").agg(
        stock_pred=("stock_prediction", "sum"),
        price_pred=("price_prediction", "mean"),
    ).reset_index()
    return agg


def prepare_comparison(df_hist, df_pred):
    df_pred = aggregate_predictions(df_pred)
    df_hist = df_hist.groupby("date_key").agg(
        stock_real=("Sum_stock_quantity", "sum"),
        price_real=("Avg_supplier_price_eur", "mean"),
    ).reset_index()
    return pd.merge(df_hist, df_pred, on="date_key", how="inner")


def plot_comparison(df, real_col, pred_col, title, ytitle):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date_key"], y=df[real_col], mode="lines+markers", name="Réel"))
    fig.add_trace(go.Scatter(x=df["date_key"], y=df[pred_col], mode="lines+markers", name="Prédit"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=400)
    return fig


def main():
    st.set_page_config(page_title="Prédictions", layout="wide")
    st.image("logo.png", width=150)
    st.title("Analyse des prédictions")

    df_hist = load_hist_data()
    tables = list_prediction_tables()
    if not tables:
        st.error("Aucune table de prédictions trouvée.")
        return

    table_name = st.selectbox("Table de prédictions", tables)
    df_pred = load_prediction_data(table_name)

    df = prepare_comparison(df_hist, df_pred)

    st.subheader("Comparaison stocks")
    fig_stock = plot_comparison(df, "stock_real", "stock_pred", "Stocks réels vs prédits", "Stock")
    st.plotly_chart(fig_stock, use_container_width=True)

    st.subheader("Comparaison prix")
    fig_price = plot_comparison(df, "price_real", "price_pred", "Prix réels vs prédits", "Prix (€)")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Données de comparaison")
    st.dataframe(df)

