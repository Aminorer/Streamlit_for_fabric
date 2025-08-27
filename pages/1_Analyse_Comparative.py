import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from constants import ASSOCIATED_COLORS
from db_utils import (
    load_hist_data,
    load_prediction_data,
    find_pred_tables,
    get_table_columns,
)
from ui_utils import (
    setup_prediction_comparison_filters,
    display_dataframe,
)


def plot_historical_vs_multi_predictions(
    hist_df: pd.DataFrame, pred_df: pd.DataFrame
) -> None:
    """Plot historical quantities against available prediction series."""
    fig = go.Figure()
    if {"date_key", "Sum_stock_quantity"}.issubset(hist_df.columns):
        fig.add_trace(
            go.Scatter(
                x=hist_df["date_key"],
                y=hist_df["Sum_stock_quantity"],
                mode="lines",
                name="Historique",
            )
        )
    for col in pred_df.columns:
        if col.startswith("stock_prediction"):
            fig.add_trace(
                go.Scatter(
                    x=pred_df["date_key"],
                    y=pred_df[col],
                    mode="lines",
                    name=col,
                )
            )
    fig.update_layout(
        title="Historique vs prédictions",
        colorway=ASSOCIATED_COLORS,
        xaxis_title="Date",
        yaxis_title="Volume",
    )
    st.plotly_chart(fig, use_container_width=True)


def analyze_prediction_accuracy_by_week(
    hist_df: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute weekly accuracy between historical and predicted values."""
    if not (
        {"date_key", "Sum_stock_quantity"}.issubset(hist_df.columns)
        and {"date_key", "stock_prediction"}.issubset(pred_df.columns)
    ):
        return pd.DataFrame()
    merged = pred_df.merge(
        hist_df[["date_key", "Sum_stock_quantity"]],
        on="date_key",
        how="left",
    )
    merged["abs_error"] = (
        merged["stock_prediction"] - merged["Sum_stock_quantity"]
    ).abs()
    merged["week"] = pd.to_datetime(merged["date_key"]).dt.to_period("W").apply(
        lambda r: r.start_time
    )
    weekly = (
        merged.groupby("week")
        .apply(
            lambda d: 1
            - d["abs_error"].sum() / d["Sum_stock_quantity"].sum()
            if d["Sum_stock_quantity"].sum()
            else 0
        )
        .reset_index(name="accuracy")
    )
    return weekly


def plot_accuracy_evolution(acc_df: pd.DataFrame) -> None:
    """Display the evolution of prediction accuracy."""
    if acc_df.empty:
        st.info("Aucune donnée de précision disponible.")
        return
    fig = px.line(
        acc_df,
        x="week",
        y="accuracy",
        markers=True,
        title="Évolution de la précision",
    )
    st.plotly_chart(fig, use_container_width=True)


def prepare_export_data(
    hist_df: pd.DataFrame, pred_df: pd.DataFrame, acc_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge historical, prediction and accuracy data for export."""
    export_df = pred_df.merge(
        hist_df,
        on=["date_key", "tyre_brand", "tyre_season_french", "tyre_fullsize"],
        how="left",
        suffixes=("_pred", "_hist"),
    )
    if not acc_df.empty:
        export_df["week"] = pd.to_datetime(export_df["date_key"]).dt.to_period("W").apply(
            lambda r: r.start_time
        )
        export_df = export_df.merge(acc_df, on="week", how="left")
    return export_df


def main() -> None:
    st.set_page_config(page_title="Analyse comparative", layout="wide")

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

    filters = setup_prediction_comparison_filters(df_full)
    try:
        df_pred = load_prediction_data(
            table_name,
            brands=filters["brands"],
            seasons=filters["seasons"],
            sizes=filters["sizes"],
            start_date=filters["start_date"],
            end_date=filters["end_date"],
        )
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors du chargement des données de prédiction : {e}")
        return

    try:
        df_hist = load_hist_data(
            brands=filters["brands"],
            seasons=filters["seasons"],
            sizes=filters["sizes"],
            start_date=filters["start_date"],
            end_date=filters["end_date"],
        )
    except Exception as e:  # pragma: no cover - Streamlit runtime
        st.error(f"Erreur lors du chargement des données historiques : {e}")
        return
    progress.progress(100)

    st.title("Analyse comparative")

    if df_pred.empty or df_hist.empty:
        st.warning("Aucune donnée disponible.")
        return

    plot_historical_vs_multi_predictions(df_hist, df_pred)
    acc_df = analyze_prediction_accuracy_by_week(df_hist, df_pred)
    plot_accuracy_evolution(acc_df)

    export_df = prepare_export_data(df_hist, df_pred, acc_df)
    st.subheader("Données combinées")
    display_dataframe(export_df)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_comparative.csv", "text/csv")


if __name__ == "__main__":
    main()

