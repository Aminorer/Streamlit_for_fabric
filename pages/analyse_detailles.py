import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analyse détaillées", layout="wide")

if "stock_data" not in st.session_state:
    st.warning("Merci de d'abord charger la table fullsize_stock_pred_enriched_xgb_june.")
    st.stop()

df = st.session_state["stock_data"].copy()

if "date_key" in df.columns:
    df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce")
    df.dropna(subset=["date_key"], inplace=True)
    df["Année"] = df["date_key"].dt.year
    df["Mois"] = df["date_key"].dt.month

years = sorted(df["Année"].dropna().unique()) if "Année" in df.columns else []
months = list(range(1, 13))
fullsizes = sorted(df["tyre_fullsize"].dropna().unique()) if "tyre_fullsize" in df.columns else []
brands = sorted(df["tyre_brand"].dropna().unique()) if "tyre_brand" in df.columns else []
seasons = sorted(df["tyre_season_french"].dropna().unique()) if "tyre_season_french" in df.columns else []

with st.sidebar.form("filtres_stock"):
    y = st.multiselect("Années", years, years)
    m = st.multiselect("Mois", months, months, format_func=lambda x: f"{x:02d}")
    fs_sel = st.multiselect("Taille", fullsizes, fullsizes)
    br_sel = st.multiselect("Marque", brands, brands)
    seas_sel = st.multiselect("Saison", seasons, seasons)
    ok = st.form_submit_button("Appliquer")

if not ok:
    st.stop()

flt = df
if years:
    flt = flt[flt["Année"].isin(y)]
if months:
    flt = flt[flt["Mois"].isin(m)]
if fs_sel:
    flt = flt[flt["tyre_fullsize"].isin(fs_sel)]
if br_sel:
    flt = flt[flt["tyre_brand"].isin(br_sel)]
if seas_sel:
    flt = flt[flt["tyre_season_french"].isin(seas_sel)]

if flt.empty:
    st.warning("Aucune donnée pour ces filtres.")
    st.stop()

st.title("Analyse détaillées du stock")

c1, c2, c3 = st.columns(3)
c1.metric("Nombre d'entrées", len(flt))
if "stock_prediction" in flt.columns:
    c2.metric("Stock moyen", f"{flt['stock_prediction'].mean():.1f}")
if "price_prediction" in flt.columns:
    c3.metric("Prix moyen", f"{flt['price_prediction'].mean():.2f}")


if {"date_key", "stock_prediction"}.issubset(flt.columns):
    fig = px.line(flt, x="date_key", y="stock_prediction", color="tyre_fullsize" if len(fs_sel) > 1 else None,
                  title="Évolution de la prédiction de stock")
    st.plotly_chart(fig, use_container_width=True)

if {"date_key", "price_prediction"}.issubset(flt.columns):
    fig = px.line(flt, x="date_key", y="price_prediction", color="tyre_fullsize" if len(fs_sel) > 1 else None,
                  title="Évolution de la prédiction de prix")
    st.plotly_chart(fig, use_container_width=True)

if {"stock_prediction", "price_prediction"}.issubset(flt.columns):
    fig = px.scatter(flt, x="stock_prediction", y="price_prediction", color="tyre_fullsize" if len(fs_sel) > 1 else None,
                     title="Prix prédits en fonction du stock")
    st.plotly_chart(fig, use_container_width=True)


cat_cols = [
    "stock_status",
    "volatility_status",
    "order_recommendation",
    "price_trend",
    "risk_level",
    "supply_chain_alert",
]
for col in cat_cols:
    if col in flt.columns:
        counts = flt[col].value_counts().reset_index()
        counts.columns = [col, "Nombre"]
        fig = px.bar(counts, x=col, y="Nombre", color=col, title=f"Répartition {col}")
        fig.update_layout(xaxis_title=col, yaxis_title="Nombre")
        st.plotly_chart(fig, use_container_width=True)


num_cols = [c for c in flt.select_dtypes(include="number").columns if c not in {"Année", "Mois"}]
for col in num_cols:
    fig = px.histogram(flt, x=col, nbins=30, title=f"Distribution de {col}")
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(flt)
