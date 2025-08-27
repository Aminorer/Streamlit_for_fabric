import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import load_prediction_data, find_pred_tables
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Analyse des prix", layout="wide")

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

    st.title("Analyse des prix")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    price_df = df.dropna(subset=["price_prediction"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_df["date_key"],
            y=price_df["price_prediction"],
            mode="lines",
            name="Prix",
        )
    )
    if "ic_price_plus" in price_df and "ic_price_minus" in price_df:
        fig.add_trace(
            go.Scatter(
                x=price_df["date_key"],
                y=price_df["ic_price_plus"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=price_df["date_key"],
                y=price_df["ic_price_minus"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(127,191,220,0.2)",
                name="Intervalle de confiance",
            )
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Alertes de saut de prix")
    if "price_jump_alert" in df:
        alerts = df[df["price_jump_alert"].notna() & (df["price_jump_alert"] != 0)]
        if alerts.empty:
            st.info("Aucune alerte détectée.")
        else:
            display_dataframe(alerts)
    else:
        st.info("Aucune alerte détectée.")

    if "margin_opportunity_days" in df:
        st.subheader("Opportunités de marge")
        marg = (
            df.groupby("tyre_brand")["margin_opportunity_days"].mean().reset_index()
        )
        marg_fig = px.bar(
            marg,
            x="tyre_brand",
            y="margin_opportunity_days",
            title="Jours d'opportunité de marge par marque",
            color_discrete_sequence=ASSOCIATED_COLORS,
        )
        st.plotly_chart(marg_fig, use_container_width=True)

    st.subheader("Données filtrées")
    display_dataframe(df)


if __name__ == "__main__":
    main()
