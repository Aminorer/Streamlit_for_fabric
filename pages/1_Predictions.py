import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ASSOCIATED_COLORS = [
    "#7fbfdc",
    "#6ba6b6",
    "#4cadb4",
    "#78b495",
    "#82b86a",
    "#45b49d",
]

from db_utils import (
    load_hist_data,
    get_engine,
)


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
    agg = (
        df.groupby("date_key")
        .agg(
            stock_pred=("stock_prediction", "sum"),
            price_pred=("price_prediction", "mean"),
        )
        .reset_index()
    )
    return agg


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


def prepare_comparison_multi(df_hist, pred_dict):
    """Return a dictionary of comparison dataframes for each prediction table."""
    out = {}
    for name, df_pred in pred_dict.items():
        out[name] = prepare_comparison(df_hist, df_pred)
    return out


def plot_comparison(df, real_col, pred_col, title, ytitle):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date_key"],
            y=df[real_col],
            mode="lines+markers",
            name="Réel",
            line=dict(color=ASSOCIATED_COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date_key"],
            y=df[pred_col],
            mode="lines+markers",
            name="Prédit",
            line=dict(color=ASSOCIATED_COLORS[1]),
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=400)
    return fig


def plot_comparison_multi(dfs, real_col, pred_col, title, ytitle):
    """Plot comparison for multiple prediction tables."""
    fig = go.Figure()
    first_df = next(iter(dfs.values()))
    fig.add_trace(
        go.Scatter(
            x=first_df["date_key"],
            y=first_df[real_col],
            mode="lines+markers",
            name="Réel",
            line=dict(color=ASSOCIATED_COLORS[0]),
        )
    )
    for idx, (name, df) in enumerate(dfs.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=df["date_key"],
                y=df[pred_col],
                mode="lines+markers",
                name=f"Prédit {name}",
                line=dict(color=ASSOCIATED_COLORS[idx % len(ASSOCIATED_COLORS)]),
            )
        )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=400)
    return fig


def plot_error(df, real_col, pred_col, title, ytitle):
    df = df.copy()
    df["error"] = df[real_col] - df[pred_col]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date_key"],
            y=df["error"],
            name="Erreur",
            marker_color=ASSOCIATED_COLORS[2],
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=400)
    return fig


def plot_error_multi(dfs, real_col, pred_col, title, ytitle):
    """Plot error for multiple tables."""
    fig = go.Figure()
    for idx, (name, df) in enumerate(dfs.items()):
        tmp = df.copy()
        tmp["error"] = tmp[real_col] - tmp[pred_col]
        fig.add_trace(
            go.Bar(
                x=tmp["date_key"],
                y=tmp["error"],
                name=name,
                marker_color=ASSOCIATED_COLORS[idx % len(ASSOCIATED_COLORS)],
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        height=400,
    )
    return fig


def plot_relative_error(df, real_col, pred_col, title):
    df = df.copy()
    df["rel_error"] = (df[pred_col] - df[real_col]) / df[real_col] * 100
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date_key"],
            y=df["rel_error"],
            name="Erreur %",
            marker_color=ASSOCIATED_COLORS[3],
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Erreur (%)", height=400
    )
    return fig


def plot_relative_error_multi(dfs, real_col, pred_col, title):
    fig = go.Figure()
    for idx, (name, df) in enumerate(dfs.items()):
        tmp = df.copy()
        tmp["rel_error"] = (tmp[pred_col] - tmp[real_col]) / tmp[real_col] * 100
        fig.add_trace(
            go.Bar(
                x=tmp["date_key"],
                y=tmp["rel_error"],
                name=name,
                marker_color=ASSOCIATED_COLORS[idx % len(ASSOCIATED_COLORS)],
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Date",
        yaxis_title="Erreur (%)",
        height=400,
    )
    return fig


def plot_abs_error(df, real_col, pred_col, title, ytitle):
    df = df.copy()
    df["abs_error"] = (df[real_col] - df[pred_col]).abs()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date_key"],
            y=df["abs_error"],
            name="Erreur absolue",
            marker_color=ASSOCIATED_COLORS[4],
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=400)
    return fig


def plot_abs_error_multi(dfs, real_col, pred_col, title, ytitle):
    fig = go.Figure()
    for idx, (name, df) in enumerate(dfs.items()):
        tmp = df.copy()
        tmp["abs_error"] = (tmp[real_col] - tmp[pred_col]).abs()
        fig.add_trace(
            go.Bar(
                x=tmp["date_key"],
                y=tmp["abs_error"],
                name=name,
                marker_color=ASSOCIATED_COLORS[idx % len(ASSOCIATED_COLORS)],
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        height=400,
    )
    return fig


def display_summary_pred(df_hist, pred_dict):
    """Display summary metrics for each prediction table."""
    if not pred_dict:
        return
    start_pred = min(df["date_key"].min() for df in pred_dict.values())

    hist_before = df_hist[df_hist["date_key"] < start_pred]
    if not hist_before.empty:
        prev = (
            hist_before.groupby("date_key")
            .agg(
                stock_real=("Sum_stock_quantity", "sum"),
                price_real=("Avg_supplier_price_eur", lambda x: x[x > 0].mean()),
            )
            .reset_index()
            .iloc[-1]
        )
        stock_real_prev = int(prev["stock_real"])
        price_real_prev = float(prev["price_real"])
    else:
        stock_real_prev = 0
        price_real_prev = 0

    col1, col2 = st.columns(2)
    col1.metric("Stock réel veille", f"{stock_real_prev:,}")
    col2.metric("Prix réel veille", f"{price_real_prev:.2f} €")

    rows = []
    for name, df_pred in pred_dict.items():
        first_pred = df_pred[df_pred["date_key"] == start_pred]
        stock_pred_first = int(first_pred["stock_prediction"].sum())
        price_pred_first = float(first_pred["price_prediction"].mean())
        rows.append(
            {
                "table": name,
                "stock_pred_first": stock_pred_first,
                "price_pred_first": price_pred_first,
            }
        )
    st.subheader("Prédictions premier jour")
    st.dataframe(pd.DataFrame(rows))


def main():
    st.set_page_config(page_title="Prédictions", layout="wide")
    st.image("logo.png", width=150)
    st.title("Analyse des prédictions")

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] * {color: black;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_hist = load_hist_cached()
    tables = list_prediction_tables()
    if not tables:
        st.error("Aucune table de prédictions trouvée.")
        return

    selected_tables = st.sidebar.multiselect("Tables de prédictions", tables, default=tables[:1])
    pred_dict = {t: load_pred_cached(t) for t in selected_tables}

    brands = st.sidebar.multiselect("Marques", sorted(df_hist["tyre_brand"].unique()))
    seasons = st.sidebar.multiselect(
        "Saisons", sorted(df_hist["tyre_season_french"].unique())
    )
    sizes = st.sidebar.multiselect("Tailles", sorted(df_hist["tyre_fullsize"].unique()))

    if st.sidebar.button("Appliquer"):
        pred_dict_f = {t: filter_data(pred, brands, seasons, sizes) for t, pred in pred_dict.items()}
        df_hist_f = filter_data(df_hist, brands, seasons, sizes)
        comps = prepare_comparison_multi(df_hist_f, pred_dict_f)

        display_summary_pred(df_hist_f, pred_dict_f)

        st.subheader("Comparaison stocks")
        fig_stock = plot_comparison_multi(
            comps, "stock_real", "stock_pred", "Stocks réels vs prédits", "Stock"
        )
        st.plotly_chart(fig_stock, use_container_width=True)
        st.subheader("Erreur de prédiction (stock)")
        fig_err_stock = plot_error_multi(
            comps, "stock_real", "stock_pred", "Erreur sur le stock", "Différence"
        )
        st.plotly_chart(fig_err_stock, use_container_width=True)

        fig_rel_stock = plot_relative_error_multi(
            comps, "stock_real", "stock_pred", "Erreur relative stock"
        )
        st.plotly_chart(fig_rel_stock, use_container_width=True)

        fig_abs_stock = plot_abs_error_multi(
            comps,
            "stock_real",
            "stock_pred",
            "Erreur absolue stock",
            "Différence absolue",
        )
        st.plotly_chart(fig_abs_stock, use_container_width=True)

        st.subheader("Comparaison prix")
        fig_price = plot_comparison_multi(
            comps, "price_real", "price_pred", "Prix réels vs prédits", "Prix (€)"
        )
        st.plotly_chart(fig_price, use_container_width=True)
        st.subheader("Erreur de prédiction (prix)")
        fig_err_price = plot_error_multi(
            comps, "price_real", "price_pred", "Erreur sur le prix", "Différence"
        )
        st.plotly_chart(fig_err_price, use_container_width=True)

        fig_rel_price = plot_relative_error_multi(
            comps, "price_real", "price_pred", "Erreur relative prix"
        )
        st.plotly_chart(fig_rel_price, use_container_width=True)

        fig_abs_price = plot_abs_error_multi(
            comps,
            "price_real",
            "price_pred",
            "Erreur absolue prix",
            "Différence absolue",
        )
        st.plotly_chart(fig_abs_price, use_container_width=True)

        frames = []
        for name, df in comps.items():
            tmp = df[["date_key", "stock_real", "stock_pred", "price_real", "price_pred"]].copy()
            tmp["table"] = name
            frames.append(tmp)
        df_display = pd.concat(frames)
        st.subheader("Données de comparaison")
        st.dataframe(df_display)
    else:
        st.info("Sélectionnez vos filtres puis cliquez sur Appliquer.")


if __name__ == "__main__":
    main()
