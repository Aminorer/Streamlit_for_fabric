import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from db_utils import (
    load_hist_data,
    get_engine,
)


def format_model_name(table_name: str) -> str:
    """Return a display friendly model name."""
    name = table_name.replace("fullsize_stock_pred_", "")
    name = name.replace("_june", "").replace("_mai", "")
    return name.upper()

ASSOCIATED_COLORS = [
    "#7fbfdc",
    "#6ba6b6",
    "#4cadb4",
    "#78b495",
    "#82b86a",
    "#45b49d",
]


@st.cache_data
def load_hist_cached():
    return load_hist_data()


@st.cache_data
def load_pred_cached(table_name):
    return load_prediction_data(table_name)


def filter_data(df, brands, seasons, sizes):
    if brands:
        df = df[df["tyre_brand"].isin(brands)]
    if seasons:
        df = df[df["tyre_season_french"].isin(seasons)]
    if sizes:
        df = df[df["tyre_fullsize"].isin(sizes)]
    return df


def list_prediction_tables(month: str):
    engine = get_engine()
    if month.lower() == "juin":
        month_filter = "AND table_name LIKE '%_june%'"
    else:
        month_filter = "AND table_name NOT LIKE '%_june%'"
    query = (
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES "
        "WHERE table_schema = 'dbo' "
        "AND table_name LIKE 'fullsize_stock_pred%' "
        f"{month_filter}"
    )
    df = pd.read_sql(query, engine)
    return df["table_name"].tolist()


def load_prediction_data(table_name):
    engine = get_engine()
    query = (
        f"SELECT date_key, tyre_brand, tyre_season_french, tyre_fullsize, "
        f"stock_prediction, price_prediction FROM dbo.{table_name}"
    )
    df = pd.read_sql(query, engine)
    df["date_key"] = pd.to_datetime(df["date_key"])
    return df


def aggregate_predictions(df):
    return (
        df.groupby("date_key")
        .agg(
            stock_pred=("stock_prediction", "sum"),
            price_pred=("price_prediction", "mean"),
        )
        .reset_index()
    )


def prepare_comparison(df_hist, df_pred):
    df_pred = aggregate_predictions(df_pred)
    df_hist = (
        df_hist.groupby("date_key")
        .agg(
            stock_real=("Sum_stock_quantity", "sum"),
            price_real=("Avg_supplier_price_eur", lambda x: x[x > 0].mean()),
        )
        .reset_index()
    )
    return pd.merge(df_hist, df_pred, on="date_key", how="inner")


def compute_metrics(df_hist, df_pred):
    df = prepare_comparison(df_hist, df_pred)
    diff = df["stock_real"] - df["stock_pred"]
    mae = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())
    mape = (diff.abs() / df["stock_real"].replace(0, np.nan)).mean() * 100
    return mae, rmse, mape


def compute_daily_summary(df_hist, pred_dict):
    """Return daily best and worst models based on MAE."""
    frames = []
    for name, df_pred in pred_dict.items():
        merged = pd.merge(
            df_hist,
            df_pred,
            on=[
                "date_key",
                "tyre_brand",
                "tyre_season_french",
                "tyre_fullsize",
            ],
            how="inner",
        )
        merged["abs_err_stock"] = (
            merged["Sum_stock_quantity"] - merged["stock_prediction"]
        ).abs()
        daily_mae = (
            merged.groupby("date_key")["abs_err_stock"].mean().reset_index()
        )
        daily_mae["table"] = name
        frames.append(daily_mae)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames)
    idx_best = all_df.groupby("date_key")["abs_err_stock"].idxmin()
    idx_worst = all_df.groupby("date_key")["abs_err_stock"].idxmax()
    best_df = all_df.loc[idx_best].rename(
        columns={"table": "best_model", "abs_err_stock": "best_mae"}
    )
    worst_df = all_df.loc[idx_worst].rename(
        columns={"table": "worst_model", "abs_err_stock": "worst_mae"}
    )
    result = pd.merge(best_df, worst_df, on="date_key")
    result = result.sort_values("date_key").head(30)
    result["best_model"] = result["best_model"].apply(format_model_name)
    result["worst_model"] = result["worst_model"].apply(format_model_name)
    return result


def plot_overall_mae(summary):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["table"],
            y=summary["mae"],
            marker_color=ASSOCIATED_COLORS[0],
        )
    )
    fig.update_layout(
        title="MAE global par modèle",
        xaxis_title="Modèle",
        yaxis_title="MAE",
        height=400,
    )
    return fig


def plot_overall_metric(summary, column, title, color_idx):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["table"],
            y=summary[column],
            marker_color=ASSOCIATED_COLORS[color_idx % len(ASSOCIATED_COLORS)],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Modèle",
        yaxis_title=column.upper(),
        height=400,
    )
    return fig


def plot_model_counts(daily, column, title, color_idx):
    counts = daily[column].value_counts()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=ASSOCIATED_COLORS[color_idx % len(ASSOCIATED_COLORS)],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Modèle",
        yaxis_title="Nombre de jours",
        height=400,
    )
    return fig


def plot_daily_best_worst(df):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date_key"],
            y=df["best_mae"],
            name="Meilleur",
            marker_color=ASSOCIATED_COLORS[1],
            hovertext=df["best_model"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["date_key"],
            y=df["worst_mae"],
            name="Pire",
            marker_color=ASSOCIATED_COLORS[2],
            hovertext=df["worst_model"],
        )
    )
    fig.update_layout(
        barmode="group",
        title="Meilleur et pire modèle par jour",
        xaxis_title="Date",
        yaxis_title="MAE",
        height=400,
    )
    return fig


def main():
    st.set_page_config(page_title="Comparatif", layout="wide")
    st.image("logo.png", width=150)
    st.title("Comparaison des modèles")

    month = st.sidebar.radio("Mois", ["Mai", "Juin"], index=0)

    df_hist = load_hist_cached()
    tables = list_prediction_tables(month)
    if not tables:
        st.error("Aucune table de prédictions trouvée.")
        return

    selected_tables = st.sidebar.multiselect(
        "Tables à comparer",
        tables,
        default=tables,
        format_func=format_model_name,
    )
    brands = st.sidebar.multiselect("Marques", sorted(df_hist["tyre_brand"].unique()))
    seasons = st.sidebar.multiselect("Saisons", sorted(df_hist["tyre_season_french"].unique()))
    sizes = st.sidebar.multiselect("Tailles", sorted(df_hist["tyre_fullsize"].unique()))

    if st.sidebar.button("Appliquer"):
        df_hist_f = filter_data(df_hist, brands, seasons, sizes)
        pred_dict = {
            t: filter_data(load_pred_cached(t), brands, seasons, sizes)
            for t in selected_tables
        }

        metrics = [compute_metrics(df_hist_f, df) for df in pred_dict.values()]
        summary = pd.DataFrame(
            {
                "table": [format_model_name(t) for t in pred_dict.keys()],
                "mae": [m[0] for m in metrics],
                "rmse": [m[1] for m in metrics],
                "mape": [m[2] for m in metrics],
            }
        )

        best_table = summary.loc[summary["mae"].idxmin(), "table"]
        st.metric("Meilleur modèle global", best_table)

        fig_mae = plot_overall_metric(summary, "mae", "MAE global par modèle", 0)
        st.plotly_chart(fig_mae, use_container_width=True)
        fig_rmse = plot_overall_metric(summary, "rmse", "RMSE global par modèle", 1)
        st.plotly_chart(fig_rmse, use_container_width=True)
        fig_mape = plot_overall_metric(summary, "mape", "MAPE global par modèle (%)", 2)
        st.plotly_chart(fig_mape, use_container_width=True)

        daily = compute_daily_summary(df_hist_f, pred_dict)
        if not daily.empty:
            fig_daily = plot_daily_best_worst(daily)
            st.plotly_chart(fig_daily, use_container_width=True)
            fig_best = plot_model_counts(daily, "best_model", "Nombre de jours meilleur", 3)
            st.plotly_chart(fig_best, use_container_width=True)
            fig_worst = plot_model_counts(daily, "worst_model", "Nombre de jours pire", 4)
            st.plotly_chart(fig_worst, use_container_width=True)
            st.subheader("Résumé quotidien")
            st.dataframe(daily)
    else:
        st.info("Sélectionnez vos filtres puis cliquez sur Appliquer.")


if __name__ == "__main__":
    main()
