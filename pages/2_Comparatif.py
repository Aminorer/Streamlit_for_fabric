import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from db_utils import (
    load_hist_data,
    get_engine,
    ensure_codex_prediction_table,
)

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


def list_prediction_tables():
    engine = get_engine()
    query = (
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES "
        "WHERE table_schema = 'dbo' AND table_name LIKE 'fullsize_stock_pred%'"
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
    return df.groupby("date_key").agg(
        stock_pred=("stock_prediction", "sum"),
        price_pred=("price_prediction", "mean"),
    ).reset_index()


def prepare_comparison(df_hist, df_pred):
    df_pred = aggregate_predictions(df_pred)
    df_hist = df_hist.groupby("date_key").agg(
        stock_real=("Sum_stock_quantity", "sum"),
        price_real=("Avg_supplier_price_eur", lambda x: x[x > 0].mean()),
    ).reset_index()
    return pd.merge(df_hist, df_pred, on="date_key", how="inner")


def compute_mae(df_hist, df_pred):
    df = prepare_comparison(df_hist, df_pred)
    df["abs_err_stock"] = (df["stock_real"] - df["stock_pred"]).abs()
    return df["abs_err_stock"].mean()


def compute_daily_summary(df_hist, tables):
    frames = []
    for t in tables:
        df_pred = load_pred_cached(t)
        comp = prepare_comparison(df_hist, df_pred)
        comp["table"] = t
        comp["abs_err_stock"] = (comp["stock_real"] - comp["stock_pred"]).abs()
        frames.append(comp[["date_key", "table", "abs_err_stock"]])
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames)
    idx_best = all_df.groupby("date_key")["abs_err_stock"].idxmin()
    idx_worst = all_df.groupby("date_key")["abs_err_stock"].idxmax()
    best_df = all_df.loc[idx_best].rename(columns={"table": "best_model", "abs_err_stock": "best_mae"})
    worst_df = all_df.loc[idx_worst].rename(columns={"table": "worst_model", "abs_err_stock": "worst_mae"})
    return pd.merge(best_df, worst_df, on="date_key")


def plot_overall_mae(summary):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["table"],
            y=summary["mae"],
            marker_color=ASSOCIATED_COLORS[0],
        )
    )
    fig.update_layout(title="MAE global par modèle", xaxis_title="Modèle", yaxis_title="MAE", height=400)
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
    ensure_codex_prediction_table(show_progress=True)
    st.image("logo.png", width=150)
    st.title("Comparaison des modèles")

    df_hist = load_hist_cached()
    tables = list_prediction_tables()
    if not tables:
        st.error("Aucune table de prédictions trouvée.")
        return

    maes = []
    for t in tables:
        df_pred = load_pred_cached(t)
        maes.append(compute_mae(df_hist, df_pred))
    summary = pd.DataFrame({"table": tables, "mae": maes})

    best_table = summary.loc[summary["mae"].idxmin(), "table"]
    st.metric("Meilleur modèle global", best_table)

    fig_overall = plot_overall_mae(summary)
    st.plotly_chart(fig_overall, use_container_width=True)

    daily = compute_daily_summary(df_hist, tables)
    if not daily.empty:
        fig_daily = plot_daily_best_worst(daily)
        st.plotly_chart(fig_daily, use_container_width=True)
        st.subheader("Résumé quotidien")
        st.dataframe(daily)


if __name__ == "__main__":
    main()
