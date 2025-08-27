import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import load_hist_data
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Comparaison historique", layout="wide")

    progress = st.progress(0)
    df = load_hist_data()
    progress.progress(100)

    _ = setup_sidebar_filters(df)
    st.title("Comparaison historique")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    history = df.groupby(["date", "category"])["stock_quantity"].sum().reset_index()
    fig = px.area(
        history,
        x="date",
        y="stock_quantity",
        color="category",
        title="Comparaison historique par catégorie",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_dataframe(history)
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "comparaison_historique.csv", "text/csv")


if __name__ == "__main__":
    main()
