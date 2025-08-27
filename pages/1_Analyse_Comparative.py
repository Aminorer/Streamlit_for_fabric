import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from sample_data import load_sample_data
from ui_utils import setup_sidebar_filters


def main() -> None:
    st.set_page_config(page_title="Analyse Comparative", layout="wide")

    with st.spinner("Chargement des données..."):
        df = load_sample_data()

    _ = setup_sidebar_filters(df)
    st.title("Analyse Comparative")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    status_counts = df["stock_status"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("OK", int(status_counts.get("OK", 0)))
    col2.metric("Rupture", int(status_counts.get("RUPTURE", 0)))
    col3.metric("Critique", int(status_counts.get("CRITIQUE", 0)))

    avg_stock = df.groupby("category")["stock_quantity"].mean().reset_index()
    bar_fig = px.bar(
        avg_stock,
        x="category",
        y="stock_quantity",
        title="Stock moyen par catégorie",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    line_df = df.groupby(["date", "category"])["stock_quantity"].sum().reset_index()
    line_fig = px.line(
        line_df,
        x="date",
        y="stock_quantity",
        color="category",
        title="Évolution par catégorie",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(line_fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_comparative.csv", "text/csv")


if __name__ == "__main__":
    main()
