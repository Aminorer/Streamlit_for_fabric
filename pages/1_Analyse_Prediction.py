import pandas as pd
import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import load_prediction_data, find_pred_tables
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Analyse prédictive", layout="wide")

    progress = st.progress(0)
    tables = find_pred_tables()
    if not tables:
        st.warning("Aucune table de prédiction disponible.")
        return
    table_name = tables[0]

    df_full = load_prediction_data(table_name)
    progress.progress(50)

    filters = setup_sidebar_filters(df_full)
    df = load_prediction_data(
        table_name,
        brands=filters["brands"],
        seasons=filters["seasons"],
        sizes=filters["sizes"],
    )
    progress.progress(100)

    st.title("Analyse prédictive")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    status_counts = (
        df["stock_status"].value_counts() if "stock_status" in df else pd.Series()
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("OK", int(status_counts.get("OK", 0)))
    col2.metric("Rupture", int(status_counts.get("RUPTURE", 0)))
    col3.metric("Critique", int(status_counts.get("CRITIQUE", 0)))

    if "criticality_score" in df:
        brand_crit = (
            df.groupby("tyre_brand")["criticality_score"].mean().reset_index()
        )
        crit_fig = px.bar(
            brand_crit,
            x="tyre_brand",
            y="criticality_score",
            title="Criticité moyenne par marque",
            color_discrete_sequence=ASSOCIATED_COLORS,
        )
        st.plotly_chart(crit_fig, use_container_width=True)

    if "stock_prediction" in df:
        evol_df = (
            df.groupby(["date_key", "tyre_brand"])["stock_prediction"].sum().reset_index()
        )
        evol_fig = px.line(
            evol_df,
            x="date_key",
            y="stock_prediction",
            color="tyre_brand",
            title="Évolution des prédictions de stock par marque",
            color_discrete_sequence=ASSOCIATED_COLORS,
        )
        st.plotly_chart(evol_fig, use_container_width=True)
    else:
        st.info("Colonne 'stock_prediction' manquante.")

    display_dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_prediction.csv", "text/csv")


if __name__ == "__main__":
    main()
