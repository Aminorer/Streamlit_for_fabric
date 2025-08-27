import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple

from constants import ASSOCIATED_COLORS
from db_utils import (
    load_prediction_data,
    find_pred_tables,
    get_table_columns,
    aggregate_predictions,
)
from ui_utils import setup_sidebar_filters, display_dataframe


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert HEX color to an RGB tuple."""
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError("Invalid HEX color")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return r, g, b


def main() -> None:
    st.set_page_config(page_title="Analyse prédictive", layout="wide")

    progress = st.progress(0)
    try:
        tables = find_pred_tables()
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors de la récupération des tables : {e}")
        return
    if not tables:
        st.warning("Aucune table de prédiction disponible.")
        return
    table_name = st.selectbox("Table de prédiction", tables)
    try:
        cols = get_table_columns(table_name)
        st.caption("Colonnes disponibles : " + ", ".join(cols))
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors de l'inspection du schéma : {e}")
        return

    try:
        df_full = load_prediction_data(table_name)
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors du chargement des données : {e}")
        return
    progress.progress(50)

    filters = setup_sidebar_filters(df_full)
    try:
        df = load_prediction_data(
            table_name,
            brands=filters["brands"],
            seasons=filters["seasons"],
            sizes=filters["sizes"],
        )
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors du chargement des données filtrées : {e}")
        return
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
        show_confidence = st.checkbox("Afficher l'intervalle de confiance")
        evol_df = aggregate_predictions(df, show_confidence=show_confidence)
        evol_fig = px.line(
            evol_df,
            x="date_key",
            y="stock_prediction",
            color="tyre_brand",
            title="Évolution des prédictions de stock par marque",
            color_discrete_sequence=ASSOCIATED_COLORS,
        )
        if show_confidence and {"ic_stock_plus", "ic_stock_minus"}.issubset(evol_df.columns):
            for idx, brand in enumerate(evol_df["tyre_brand"].unique()):
                brand_df = evol_df[evol_df["tyre_brand"] == brand]
                r, g, b = hex_to_rgb(ASSOCIATED_COLORS[idx % len(ASSOCIATED_COLORS)])
                evol_fig.add_trace(
                    go.Scatter(
                        x=brand_df["date_key"],
                        y=brand_df["ic_stock_plus"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                evol_fig.add_trace(
                    go.Scatter(
                        x=brand_df["date_key"],
                        y=brand_df["ic_stock_minus"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba({r},{g},{b},0.2)",
                        name=f"IC {brand}",
                    )
                )
        st.plotly_chart(evol_fig, use_container_width=True)
    else:
        st.info("Colonne 'stock_prediction' manquante.")

    display_dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_prediction.csv", "text/csv")


if __name__ == "__main__":
    main()
