import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db_utils import get_engine

ASSOCIATED_COLORS = [
    "#7fbfdc",
    "#6ba6b6",
    "#4cadb4",
    "#78b495",
    "#82b86a",
    "#45b49d",
]

def format_model_name(name: str) -> str:
    name = name.replace("fullsize_stock_pred_", "")
    name = name.replace("_june", "").replace("_mai", "")
    return name.upper()

@st.cache_data
def list_prediction_tables():
    engine = get_engine()
    query = (
        "SELECT table_name FROM INFORMATION_SCHEMA.TABLES "
        "WHERE table_schema = 'dbo' "
        "AND table_name LIKE 'fullsize_stock_pred%' "
        "AND table_name LIKE '%_june%'"
    )
    df = pd.read_sql(query, engine)
    return df["table_name"].tolist()

@st.cache_data
def load_prediction_data(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM dbo.{table_name}", engine)
    if "date_key" in df.columns:
        df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce")
        df.dropna(subset=["date_key"], inplace=True)
    return df

def filter_data(df: pd.DataFrame, brands, seasons, sizes) -> pd.DataFrame:
    if brands:
        df = df[df["tyre_brand"].isin(brands)]
    if seasons:
        df = df[df["tyre_season_french"].isin(seasons)]
    if sizes:
        df = df[df["tyre_fullsize"].isin(sizes)]
    return df

st.set_page_config(page_title="Analyse détaillée", layout="wide")
st.image("logo.png", width=150)
st.title("Analyse détaillée des prédictions")

st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] * {color: black;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Sidebar filters -----
tables = list_prediction_tables()
if not tables:
    st.error("Aucune table de prédictions trouvée.")
    st.stop()

selected_tables = st.sidebar.multiselect(
    "Tables de prédictions",
    tables,
    default=tables[:1],
    format_func=format_model_name,
)

if not selected_tables:
    st.info("Sélectionnez au moins une table.")
    st.stop()

pred_dict = {t: load_prediction_data(t) for t in selected_tables}
all_df = pd.concat(pred_dict.values(), ignore_index=True)

brands = st.sidebar.multiselect("Marques", sorted(all_df["tyre_brand"].dropna().unique()))
seasons = st.sidebar.multiselect("Saisons", sorted(all_df["tyre_season_french"].dropna().unique()))
sizes = st.sidebar.multiselect("Tailles", sorted(all_df["tyre_fullsize"].dropna().unique()))

if not st.sidebar.button("Appliquer"):
    st.stop()

flt = filter_data(all_df, brands, seasons, sizes)
if flt.empty:
    st.warning("Aucune donnée pour ces filtres.")
    st.stop()

# Aggregate once for faster plots
agg_map = {}
for c in ["stock_prediction", "ic_stock_plus", "ic_stock_minus"]:
    if c in flt.columns:
        agg_map[c] = "sum"
for c in ["price_prediction", "ic_price_plus", "ic_price_minus", "stability_index"]:
    if c in flt.columns:
        agg_map[c] = "mean"
if agg_map:
    agg_df = flt.groupby("date_key").agg(agg_map).reset_index()
else:
    agg_df = pd.DataFrame()

# ----- Metrics -----
c1, c2, c3 = st.columns(3)
c1.metric("Nombre d'entrées", len(flt))
if "stock_prediction" in flt.columns:
    c2.metric("Stock moyen", f"{flt['stock_prediction'].mean():.1f}")
if "price_prediction" in flt.columns:
    c3.metric("Prix moyen", f"{flt['price_prediction'].mean():.2f} €")

c4, c5 = st.columns(2)
if "stock_prediction" in flt.columns:
    c4.metric(
        "Stock total prédicté", f"{int(flt['stock_prediction'].sum()):,}"
    )
if "price_prediction" in flt.columns:
    c5.metric("Prix médian", f"{flt['price_prediction'].median():.2f} €")

# ----- Graphs -----

# 1. Stock prediction over time
if {"date_key", "stock_prediction"}.issubset(agg_df.columns):
    fig = px.line(
        agg_df,
        x="date_key",
        y="stock_prediction",
        title="Évolution des prédictions de stock",
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. Price prediction over time
if {"date_key", "price_prediction"}.issubset(agg_df.columns):
    fig = px.line(
        agg_df,
        x="date_key",
        y="price_prediction",
        title="Évolution des prédictions de prix",
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Stock prediction confidence band
if {"date_key", "stock_prediction", "ic_stock_plus", "ic_stock_minus"}.issubset(agg_df.columns):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["stock_prediction"],
            name="Prédiction",
            line=dict(color=ASSOCIATED_COLORS[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["ic_stock_plus"],
            name="IC +",
            line=dict(color=ASSOCIATED_COLORS[1], dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["ic_stock_minus"],
            name="IC -",
            line=dict(color=ASSOCIATED_COLORS[1], dash="dash"),
            fill="tonexty",
        )
    )
    fig.update_layout(
        title="Bande de confiance sur le stock",
        xaxis_title="Date",
        yaxis_title="Stock",
    )
    st.plotly_chart(fig, use_container_width=True)

# 4. Price prediction confidence band
if {"date_key", "price_prediction", "ic_price_plus", "ic_price_minus"}.issubset(agg_df.columns):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["price_prediction"],
            name="Prédiction",
            line=dict(color=ASSOCIATED_COLORS[2]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["ic_price_plus"],
            name="IC +",
            line=dict(color=ASSOCIATED_COLORS[3], dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg_df["date_key"],
            y=agg_df["ic_price_minus"],
            name="IC -",
            line=dict(color=ASSOCIATED_COLORS[3], dash="dash"),
            fill="tonexty",
        )
    )
    fig.update_layout(
        title="Bande de confiance sur le prix",
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig, use_container_width=True)

# 5. Mean confidence by brand
if {"tyre_brand", "prediction_confidence"}.issubset(flt.columns):
    conf = flt.groupby("tyre_brand")["prediction_confidence"].mean().reset_index()
    fig = px.bar(conf, x="tyre_brand", y="prediction_confidence", title="Indice de confiance moyen par marque")
    st.plotly_chart(fig, use_container_width=True)

# 6. Stock status distribution
if "stock_status" in flt.columns:
    counts = flt["stock_status"].value_counts().reset_index()
    counts.columns = ["stock_status", "count"]
    fig = px.bar(counts, x="stock_status", y="count", title="Distribution du statut de stock")
    st.plotly_chart(fig, use_container_width=True)

# 7. Volatility status distribution
if "volatility_status" in flt.columns:
    counts = flt["volatility_status"].value_counts().reset_index()
    counts.columns = ["volatility_status", "count"]
    fig = px.bar(counts, x="volatility_status", y="count", title="Distribution du statut de volatilité")
    st.plotly_chart(fig, use_container_width=True)

# 8. Out of stock days by model
if {"tyre_fullsize", "out_of_stock_days"}.issubset(flt.columns):
    oos = flt.groupby("tyre_fullsize")["out_of_stock_days"].sum().reset_index()
    fig = px.bar(oos, x="tyre_fullsize", y="out_of_stock_days", title="Nombre de jours sans stock par modèle")
    st.plotly_chart(fig, use_container_width=True)

# 9. Maximum rupture duration by model
if {"tyre_fullsize", "rupture_duration_max"}.issubset(flt.columns):
    rup = flt.groupby("tyre_fullsize")["rupture_duration_max"].max().reset_index()
    fig = px.bar(rup, x="tyre_fullsize", y="rupture_duration_max", title="Durée maximale de rupture par modèle")
    st.plotly_chart(fig, use_container_width=True)

# 10. Stability index over time
if {"date_key", "stability_index"}.issubset(agg_df.columns):
    fig = px.line(
        agg_df,
        x="date_key",
        y="stability_index",
        title="Évolution du stability_index",
    )
    st.plotly_chart(fig, use_container_width=True)

# 11. Heatmap brand vs volatility type with criticality score
if {"tyre_brand", "volatility_type", "criticality_score"}.issubset(flt.columns):
    heat = flt.pivot_table(index="tyre_brand", columns="volatility_type", values="criticality_score", aggfunc="mean")
    fig = px.imshow(heat, title="Criticality score par marque et type de volatilité")
    st.plotly_chart(fig, use_container_width=True)

# 12. Histogram of anomaly alerts
if "anomaly_alert" in flt.columns:
    fig = px.histogram(flt, x="anomaly_alert", title="Nombre d'alertes anomalies")
    st.plotly_chart(fig, use_container_width=True)

# 13. Risk level distribution per brand
if {"tyre_brand", "risk_level"}.issubset(flt.columns):
    risk = flt.groupby(["tyre_brand", "risk_level"]).size().reset_index(name="count")
    fig = px.bar(risk, x="tyre_brand", y="count", color="risk_level", barmode="group", title="Risk level par marque")
    st.plotly_chart(fig, use_container_width=True)

# 14. Margin opportunity days distribution
if "margin_opportunity_days" in flt.columns:
    fig = px.histogram(flt, x="margin_opportunity_days", nbins=30, title="Distribution du margin_opportunity_days")
    st.plotly_chart(fig, use_container_width=True)

# 15. Timeline of main rupture and last safe order dates
if {"tyre_fullsize", "main_rupture_date", "last_safe_order_date"}.issubset(flt.columns):
    tmp = flt[["tyre_fullsize", "main_rupture_date", "last_safe_order_date"]].dropna()
    tmp["main_rupture_date"] = pd.to_datetime(tmp["main_rupture_date"], errors="coerce")
    tmp["last_safe_order_date"] = pd.to_datetime(tmp["last_safe_order_date"], errors="coerce")
    fig = px.timeline(tmp, x_start="main_rupture_date", x_end="last_safe_order_date", y="tyre_fullsize",
                      title="Périodes de rupture par modèle")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# Display filtered dataframe
st.subheader("Données filtrées")
st.dataframe(flt)
