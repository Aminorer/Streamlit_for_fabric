import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import (
    load_prediction_data,
    find_pred_tables,
    get_table_columns,
)
from ui_utils import setup_sidebar_filters, display_dataframe


def main() -> None:
    st.set_page_config(page_title="Volatilité & Risques", layout="wide")

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

    st.title("Volatilité & Risques")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    if "volatility_type" in df and "volatility_status" in df:
        vol_rank = (
            df.groupby(["volatility_type", "volatility_status"]).size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig = px.bar(
            vol_rank,
            x="volatility_type",
            y="count",
            color="volatility_status",
            title="Classement par type et statut de volatilité",
            color_discrete_sequence=ASSOCIATED_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)
        display_dataframe(vol_rank)

    if "risk_level" in df:
        risk_agg = (
            df.groupby(["tyre_brand", "tyre_season_french", "risk_level"]).size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.subheader("Agrégation des niveaux de risque par marque et saison")
        display_dataframe(risk_agg)

    st.subheader("Alertes")
    alert_mask = False
    if "anomaly_alert" in df:
        alert_mask |= df["anomaly_alert"].notna() & (df["anomaly_alert"] != "NONE")
    if "supply_chain_alert" in df:
        alert_mask |= df["supply_chain_alert"].notna() & (df["supply_chain_alert"] != "NONE")
    alerts = df[alert_mask] if isinstance(alert_mask, bool) else df[alert_mask]
    if alerts.empty:
        st.info("Aucune alerte détectée.")
    else:
        display_dataframe(alerts)

    st.subheader("Données filtrées")
    display_dataframe(df)


if __name__ == "__main__":
    main()
