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
    hex_to_rgb,
)


def plot_historical_vs_multi_predictions(
    hist_df: pd.DataFrame, pred_df: pd.DataFrame, use_emoji: bool = True
) -> None:
    """Plot historical quantities against available prediction series.

    Icons are added to trace names for clarity; set ``use_emoji`` to ``False``
    to disable them.
    """
    fig = go.Figure()
    color_idx = 0
    hist_name = "📈 Historique" if use_emoji else "Historique"
    if {"date_key", "Sum_stock_quantity"}.issubset(hist_df.columns):
        fig.add_trace(
            go.Scatter(
                x=hist_df["date_key"],
                y=hist_df["Sum_stock_quantity"],
                mode="lines",
                name=hist_name,
                line=dict(color=ASSOCIATED_COLORS[color_idx]),
            )
        )
        color_idx += 1
    pred_cols = [c for c in pred_df.columns if c.startswith("stock_prediction")]
    pred_template = "🔮 Prédiction{}" if use_emoji else "Prédiction{}"
    ic_template = "🎯 IC Prédiction{}" if use_emoji else "IC Prédiction{}"
    for idx, col in enumerate(pred_cols):
        color = ASSOCIATED_COLORS[(idx + color_idx) % len(ASSOCIATED_COLORS)]
        suffix = col[len("stock_prediction") :]
        suffix_label = suffix.replace("_", " ")
        fig.add_trace(
            go.Scatter(
                x=pred_df["date_key"],
                y=pred_df[col],
                mode="lines",
                name=pred_template.format(suffix_label),
                line=dict(color=color),
            )
        )
        ic_plus = f"ic_stock_plus{suffix}"
        ic_minus = f"ic_stock_minus{suffix}"
        if {ic_plus, ic_minus}.issubset(pred_df.columns):
            r, g, b = hex_to_rgb(color)
            fig.add_trace(
                go.Scatter(
                    x=pred_df["date_key"],
                    y=pred_df[ic_plus],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pred_df["date_key"],
                    y=pred_df[ic_minus],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    name=ic_template.format(suffix_label),
                )
            )
    fig.update_layout(
        title="Historique vs prédictions",
        xaxis_title="Date",
        yaxis_title="Volume",
    )
    st.plotly_chart(fig, use_container_width=True)


def analyze_prediction_accuracy_by_week(
    hist_df: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute weekly prediction accuracy.

    Rows where ``Sum_stock_quantity`` equals ``0`` are excluded before
    calculating the Mean Absolute Percentage Error (MAPE) to avoid division by
    zero. Weekly accuracy is returned as ``1 - MAPE``.
    """
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
    # Exclude rows with zero historical quantity to avoid divide-by-zero in MAPE
    merged = merged[merged["Sum_stock_quantity"] != 0].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["abs_perc_error"] = (
        (merged["stock_prediction"] - merged["Sum_stock_quantity"]).abs()
        / merged["Sum_stock_quantity"]
    )
    merged["week"] = pd.to_datetime(merged["date_key"]).dt.to_period("W").apply(
        lambda r: r.start_time
    )
    weekly = (
        1
        - merged.groupby("week")["abs_perc_error"].mean()
    ).reset_index(name="accuracy")
    return weekly


def plot_accuracy_evolution(acc_df: pd.DataFrame) -> None:
    """Display the evolution of prediction accuracy."""
    if acc_df.empty:
        st.info("Aucune donnée de précision disponible.")
        return
    # ``week`` can contain localized labels (e.g. ``"01/01/2024 - Semaine 1"``).
    # Convert them back to ``datetime`` for proper chronological sorting,
    # then plot using the original labels to preserve the display format.
    df = acc_df.copy()
    df["_week_dt"] = pd.to_datetime(
        df["week"].astype(str).str.split(" - ").str[0],
        errors="coerce",
        dayfirst=True,
    )
    df = df.sort_values("_week_dt")
    fig = px.line(
        df,
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

    use_emoji = st.checkbox("Afficher des icônes dans la légende", value=True)
    plot_historical_vs_multi_predictions(df_hist, df_pred, use_emoji=use_emoji)
    acc_df = analyze_prediction_accuracy_by_week(df_hist, df_pred)
    plot_accuracy_evolution(acc_df)

    export_df = prepare_export_data(df_hist, df_pred, acc_df)
    st.subheader("Données combinées")
    display_dataframe(export_df)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Exporter CSV", csv, "analyse_comparative.csv", "text/csv")


if __name__ == "__main__":
    main()

