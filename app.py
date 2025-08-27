import pandas as pd
import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import load_hist_data, load_prediction_data
from ui_utils import setup_sidebar_filters, display_dataframe


st.set_page_config(page_title="Tableau de bord exécutif", layout="wide")

filters = setup_sidebar_filters()

progress = st.progress(0)
df_hist = load_hist_data(
    brands=filters["brands"],
    seasons=filters["seasons"],
    sizes=filters["sizes"],
)
df_pred = load_prediction_data(
    brands=filters["brands"],
    seasons=filters["seasons"],
    sizes=filters["sizes"],
)
progress.progress(100)

df = df_hist if filters["activity_type"] == "Historique" else df_pred

st.title("Tableau de bord exécutif")

if df.empty:
    st.warning("Aucune donnée disponible.")
else:
    status_counts = df["stock_status"].value_counts()
    critical_items = int(status_counts.get("CRITIQUE", 0))
    next_rupture = pd.to_datetime(df["main_rupture_date"]).min()
    avg_criticality = float(df["criticality_score"].mean())

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Articles critiques", f"{critical_items}")
    kpi2.metric(
        "Prochaine rupture",
        next_rupture.strftime("%Y-%m-%d") if pd.notna(next_rupture) else "N/A",
    )
    kpi3.metric("Criticité moyenne", f"{avg_criticality:.2f}")

    status_df = status_counts.reset_index()
    status_df.columns = ["stock_status", "count"]
    fig_status = px.bar(
        status_df,
        x="stock_status",
        y="count",
        title="Répartition par statut de stock",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(fig_status, use_container_width=True)

    display_dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les données", csv, "dashboard.csv", "text/csv")
