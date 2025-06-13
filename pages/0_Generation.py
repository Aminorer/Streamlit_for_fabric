import streamlit as st
import pandas as pd

from db_utils import load_hist_data, generate_codex_predictions, save_dataframe_to_table


def main():
    st.set_page_config(page_title="Générer des prédictions", layout="wide")
    st.image("logo.png", width=150)
    st.title("Génération d'une nouvelle table de prédictions")

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] * {color: black;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Lancer la génération"):
        df_hist = load_hist_data()
        progress = st.progress(0.0)
        preds = generate_codex_predictions(
            df_hist,
            progress_callback=lambda v: progress.progress(v),
        )
        table_name = f"fullsize_stock_pred_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        save_dataframe_to_table(preds, table_name)
        progress.empty()
        st.success(f"Table {table_name} générée")


if __name__ == "__main__":
    main()
