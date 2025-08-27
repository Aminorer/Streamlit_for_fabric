import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from sample_data import load_sample_data


def main() -> None:
    st.set_page_config(page_title="Analyse détaillée", layout="wide")
    st.title("Analyse détaillée")

    with st.spinner("Chargement des données..."):
        df = load_sample_data()

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    statuses = ["TOUS"] + sorted(df["stock_status"].unique())
    status = st.selectbox("Statut", statuses)
    if status != "TOUS":
        df = df[df["stock_status"] == status]

    earliest = df["main_rupture_date"].min()
    avg_crit = float(df["criticality_score"].mean())
    total_stock = int(df["stock_quantity"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Date de rupture min", str(earliest.date()))
    k2.metric("Criticité moyenne", f"{avg_crit:.2f}")
    k3.metric("Stock total", f"{total_stock}")

    fig = px.scatter(
        df,
        x="date",
        y="criticality_score",
        color="category",
        title="Criticité dans le temps",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_detaillee.csv", "text/csv")


if __name__ == "__main__":
    main()
