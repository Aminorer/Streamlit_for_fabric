import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from sample_data import load_sample_data


def main() -> None:
    st.set_page_config(page_title="Analyse par catégorie", layout="wide")
    st.title("Analyse par catégorie")

    with st.spinner("Chargement des données..."):
        df = load_sample_data()

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    categories = sorted(df["category"].unique())
    category = st.selectbox("Catégorie", categories)
    subset = df[df["category"] == category]

    total_stock = int(subset["stock_quantity"].sum())
    criticality = float(subset["criticality_score"].mean())
    critical_items = int((subset["stock_status"] == "CRITIQUE").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Stock total", f"{total_stock}")
    col2.metric("Criticité moyenne", f"{criticality:.2f}")
    col3.metric("Articles critiques", f"{critical_items}")

    line_fig = px.line(
        subset,
        x="date",
        y="stock_quantity",
        title=f"Évolution du stock - Catégorie {category}",
        color_discrete_sequence=[ASSOCIATED_COLORS[0]],
    )
    st.plotly_chart(line_fig, use_container_width=True)

    st.dataframe(subset, use_container_width=True)
    csv = subset.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, f"categorie_{category}.csv", "text/csv")


if __name__ == "__main__":
    main()
