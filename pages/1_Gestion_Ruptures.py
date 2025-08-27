import plotly.express as px
import streamlit as st

from db_utils import load_prediction_data, find_pred_tables
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Gestion des ruptures", layout="wide")

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

    st.title("Gestion des ruptures")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    rupture_df = (
        df.dropna(subset=["main_rupture_date"])  # type: ignore[arg-type]
        .groupby("main_rupture_date")
        .size()
        .reset_index(name="ruptures")
    )
    if not rupture_df.empty:
        fig = px.bar(
            rupture_df,
            x="main_rupture_date",
            y="ruptures",
            title="Timeline des ruptures",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune rupture détectée.")

    st.subheader("Produits critiques")
    critical_df = df.sort_values("criticality_score", ascending=False)
    display_dataframe(critical_df)

    st.subheader("Recommandations d'achat")
    recomm_df = (
        df[["tyre_brand", "tyre_fullsize", "recommended_volume"]]
        .sort_values("recommended_volume", ascending=False)
    )
    display_dataframe(recomm_df)

    st.subheader("Données filtrées")
    display_dataframe(df)


if __name__ == "__main__":
    main()
