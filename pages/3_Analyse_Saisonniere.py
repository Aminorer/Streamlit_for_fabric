import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from sample_data import load_sample_data
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Analyse saisonnière", layout="wide")

    progress = st.progress(0)
    df = load_sample_data()
    progress.progress(100)

    _ = setup_sidebar_filters(df)
    st.title("Analyse saisonnière")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    monthly_summary = monthly.groupby("month")["stock_quantity"].sum().reset_index()
    fig = px.line(
        monthly_summary,
        x="month",
        y="stock_quantity",
        title="Stock total par mois",
        color_discrete_sequence=[ASSOCIATED_COLORS[0]],
    )
    st.plotly_chart(fig, use_container_width=True)

    display_dataframe(monthly)
    csv = monthly.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_saisonniere.csv", "text/csv")


if __name__ == "__main__":
    main()
