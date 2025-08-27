import pandas as pd
import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import load_hist_data
from ui_utils import setup_sidebar_filters, display_dataframe


st.set_page_config(page_title="Tableau de bord exécutif", layout="wide")

progress = st.progress(0)
df = load_hist_data()
progress.progress(100)

_ = setup_sidebar_filters(df)
st.title("Tableau de bord exécutif")

if df.empty:
    st.warning("Aucune donnée disponible.")
else:
    total_stock = int(df["stock_quantity"].sum())
    critical_items = int((df["stock_status"] == "CRITIQUE").sum())
    avg_criticality = float(df["criticality_score"].mean())

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Stock total", f"{total_stock}")
    kpi2.metric("Articles critiques", f"{critical_items}")
    kpi3.metric("Criticité moyenne", f"{avg_criticality:.2f}")

    daily_stock = df.groupby("date")["stock_quantity"].sum().reset_index()
    fig_stock = px.line(
        daily_stock,
        x="date",
        y="stock_quantity",
        title="Évolution du stock",
    )
    fig_stock.update_traces(line_color=ASSOCIATED_COLORS[0])
    st.plotly_chart(fig_stock, use_container_width=True)

    crit_by_cat = df.groupby("category")["criticality_score"].mean().reset_index()
    fig_cat = px.bar(
        crit_by_cat,
        x="category",
        y="criticality_score",
        title="Criticité moyenne par catégorie",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    display_dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les données", csv, "dashboard.csv", "text/csv")
