import streamlit as st
import pyodbc
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path="secret.env")

def get_engine():

    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    driver = os.getenv("SQL_DRIVER")
    if driver is None:
        raise ValueError("La variable d'environnement SQL_DRIVER est manquante.")
    driver = driver.replace(" ", "+")
  

    connection_string = (
        f"mssql+pyodbc://{user}:{password}@{server}:1433/{database}"
        f"?driver={driver}"
        "&authentication=ActiveDirectoryPassword"
        "&encrypt=yes"
        "&TrustServerCertificate=no"
    )
    return create_engine(connection_string)

def load_table(table_name):
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)

def plot_model_prediction(model_name, hist_df, pred_df):
    df = pred_df.merge(
        hist_df,
        on=['date', 'tyre_fullsize', 'tyre_brand', 'tyre_season_french'],
        suffixes=('_prediction', '_quantity')
    )
    if df.empty:
        return None
    rmse = mean_squared_error(df['stock_quantity'], df['stock_prediction'], squared=False)
    mae = mean_absolute_error(df['stock_quantity'], df['stock_prediction'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['stock_quantity'], mode='lines', name='Stock réel', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['stock_prediction'], mode='lines', name='Stock prédit', line=dict(color='blue')))
    fig.update_layout(title=f"{model_name} — RMSE: {rmse:.0f} | MAE: {mae:.0f}", xaxis_title="Date", yaxis_title="Stock", height=300)
    return fig

def get_all_predicted_couples(model_tables):
    engine = get_engine()
    all_couples = pd.DataFrame()
    for table in model_tables:
        try:
            df = pd.read_sql(f"SELECT DISTINCT tyre_fullsize, tyre_brand, tyre_season_french FROM dbo.{table}", engine)
            all_couples = pd.concat([all_couples, df], ignore_index=True)
        except Exception as e:
            st.sidebar.warning(f"Erreur chargement {table} : {e}")
    all_couples.drop_duplicates(inplace=True)
    return all_couples

def show_available_couples(model_tables):
    st.sidebar.markdown("### 🔍 Couples prédits par modèle")
    for table in model_tables:
        try:
            df = load_table(f"dbo.{table}")
            couples = df[['tyre_fullsize', 'tyre_brand', 'tyre_season_french']].drop_duplicates()
            with st.sidebar.expander(f"🧪 {table}", expanded=False):
                st.write(f"{len(couples)} couples disponibles")
                st.dataframe(couples)
        except Exception as e:
            st.sidebar.error(f"Erreur {table} : {e}")

def main():
    st.set_page_config(page_title="Comparaison modèles stock", layout="wide")
    st.title("📦 Comparaison des prédictions de stock par modèle")

    model_tables = [
        'fullsize_stock_pred_lgb', 'fullsize_stock_pred_lstm',
        'fullsize_stock_pred_tcn', 'fullsize_stock_pred_xgb',
        'fullsize_stock_pred_gbt', 'fullsize_stock_pred_sarimax',
        'fullsize_stock_pred_prophet', 'fullsize_stock_pred_mlp',
        'fullsize_stock_pred_linreg'
    ]

    with st.spinner("Chargement des données..."):
        all_predicted_couples = get_all_predicted_couples(model_tables)
        if all_predicted_couples.empty:
            st.sidebar.error("❌ Aucun couple disponible dans les tables de prédictions.")
            st.stop()

        selected_row = st.sidebar.selectbox(
            "🧩 Choisir un couple (Fullsize / Brand / Saison)",
            all_predicted_couples.apply(lambda row: f"{row['tyre_fullsize']} | {row['tyre_brand']} | {row['tyre_season_french']}", axis=1)
        )
        if not selected_row:
            st.sidebar.warning("Sélection invalide.")
            st.stop()

        selected_fullsize, selected_brand, selected_season = selected_row.split(" | ")

        hist_df = load_table("dbo.df_stock_hist")
        hist_df = hist_df[['date_key', 'tyre_fullsize', 'tyre_brand', 'tyre_season_french', 'Sum_stock_quantity']]
        hist_df.columns = ['date', 'tyre_fullsize', 'tyre_brand', 'tyre_season_french', 'stock_quantity']
        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.date

        filtered_hist = hist_df[
            (hist_df['tyre_fullsize'].str.strip().str.lower() == selected_fullsize.strip().lower()) &
            (hist_df['tyre_brand'].str.strip().str.lower() == selected_brand.strip().lower()) &
            (hist_df['tyre_season_french'].str.strip().str.lower() == selected_season.strip().lower())
        ]

        min_date, max_date = filtered_hist['date'].min(), filtered_hist['date'].max()
        selected_range = st.sidebar.slider("📅 Plage de dates", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        filtered_hist = filtered_hist[(filtered_hist['date'] >= selected_range[0]) & (filtered_hist['date'] <= selected_range[1])]

        show_available_couples(model_tables)

        figs = []
        for table in model_tables:
            df_pred = load_table(f"dbo.{table}")
            df_pred = df_pred[['date_key', 'tyre_fullsize', 'tyre_brand', 'tyre_season_french', 'stock_prediction']]
            df_pred.columns = ['date', 'tyre_fullsize', 'tyre_brand', 'tyre_season_french', 'stock_prediction']
            df_pred['date'] = pd.to_datetime(df_pred['date']).dt.date

            df_pred = df_pred[
                (df_pred['tyre_fullsize'].str.strip().str.lower() == selected_fullsize.strip().lower()) &
                (df_pred['tyre_brand'].str.strip().str.lower() == selected_brand.strip().lower()) &
                (df_pred['tyre_season_french'].str.strip().str.lower() == selected_season.strip().lower())
            ]
            df_pred = df_pred[(df_pred['date'] >= selected_range[0]) & (df_pred['date'] <= selected_range[1])]

            st.write(f"📈 {table} : {len(df_pred)} lignes après filtre")

            if not df_pred.empty and not filtered_hist.empty:
                fig = plot_model_prediction(table.upper().replace('FULLSIZE_STOCK_PRED_', ''), filtered_hist, df_pred)
                if fig:
                    figs.append(fig)

    cols = st.columns(3)
    for i, fig in enumerate(figs):
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
