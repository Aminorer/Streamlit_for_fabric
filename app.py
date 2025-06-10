import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db_utils import load_hist_data


@st.cache_data
def load_hist_cached():
    return load_hist_data()


def filter_data(df, brands, seasons, sizes):
    if brands:
        df = df[df["tyre_brand"].isin(brands)]
    if seasons:
        df = df[df["tyre_season_french"].isin(seasons)]
    if sizes:
        df = df[df["tyre_fullsize"].isin(sizes)]
    return df


def plot_stock_by_brand(df):
    daily_brand = (
        df.groupby(["date_key", "tyre_brand"])["Sum_stock_quantity"].sum().reset_index()
    )
    fig = go.Figure()
    for brand in daily_brand["tyre_brand"].unique():
        data = daily_brand[daily_brand["tyre_brand"] == brand]
        fig.add_trace(
            go.Scatter(
                x=data["date_key"],
                y=data["Sum_stock_quantity"],
                mode="lines",
                stackgroup="one",
                name=brand,
                line=dict(color="#001944"),
            )
        )
    fig.update_layout(title="Stock par marque", xaxis_title="Date", yaxis_title="Stock", height=400)
    return fig


def display_summary(df):
    total_stock = int(df["Sum_stock_quantity"].sum())
    avg_price = float(df["Avg_supplier_price_eur"].mean())
    col1, col2 = st.columns(2)
    col1.metric("Stock total", f"{total_stock:,}")
    col2.metric("Prix moyen", f"{avg_price:.2f} €")

def plot_stock_sum(df):
    daily_stock = df.groupby("date_key")["Sum_stock_quantity"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stock["date_key"],
        y=daily_stock["Sum_stock_quantity"],
        mode="lines+markers",
        name="Stock total",
        line=dict(color="#001944"),
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
        line=dict(color="#001944"),
    ))
    fig.update_layout(title="Évolution du prix moyen fournisseur", xaxis_title="Date", yaxis_title="Prix (€)", height=400)
    return fig

def main():
    st.set_page_config(page_title="Historique des stocks", layout="wide")
    st.image("logo.png", width=150)
    st.title("Historique des stocks")

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] * {color: black;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_hist_cached()

    if df.empty:
        st.error("Aucune donnée disponible.")
        return

    brands = st.sidebar.multiselect("Marques", sorted(df["tyre_brand"].unique()))
    seasons = st.sidebar.multiselect("Saisons", sorted(df["tyre_season_french"].unique()))
    sizes = st.sidebar.multiselect("Tailles", sorted(df["tyre_fullsize"].unique()))

    if st.sidebar.button("Appliquer"):
        df = filter_data(df, brands, seasons, sizes)

        display_summary(df)

        fig_stock = plot_stock_sum(df)
        st.plotly_chart(fig_stock, use_container_width=True)

        fig_price = plot_price_avg(df)
        st.plotly_chart(fig_price, use_container_width=True)

        fig_brand = plot_stock_by_brand(df)
        st.plotly_chart(fig_brand, use_container_width=True)
    else:
        st.info("Sélectionnez vos filtres puis cliquez sur Appliquer.")

if __name__ == "__main__":
    main()
