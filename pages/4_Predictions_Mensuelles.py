import pandas as pd
import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from sample_data import load_sample_data
from ui_utils import setup_sidebar_filters


def main() -> None:
    st.set_page_config(page_title="Prédictions mensuelles", layout="wide")

    with st.spinner("Chargement des données..."):
        df = load_sample_data()

    _ = setup_sidebar_filters(df)
    st.title("Prédictions mensuelles")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    daily = df.groupby("date")["stock_quantity"].sum().reset_index()
    daily["prediction"] = daily["stock_quantity"].rolling(7, min_periods=1).mean()

    fig = px.line(
        daily,
        x="date",
        y=["stock_quantity", "prediction"],
        labels={"value": "Stock", "variable": "Série"},
        title="Stock réel vs prédit",
        color_discrete_sequence=ASSOCIATED_COLORS[:2],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(daily, use_container_width=True)
    csv = daily.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "predictions_mensuelles.csv", "text/csv")


if __name__ == "__main__":
    main()
