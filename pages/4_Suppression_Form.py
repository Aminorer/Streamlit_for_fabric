import pandas as pd
import streamlit as st


def main() -> None:
    """Demonstrate batch deletion using st.form with st.data_editor.

    This example avoids rerunning the app on every checkbox change by
    wrapping the table in a form. Selections are applied only when the user
    submits the form.
    """
    st.title("Gestion de suppression par lot")

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(
            {
                "Token": ["ORG_10", "ORG_11", "ORG_12"],
                "Type": ["ORG", "ORG", "ORG"],
                "Occurrences": [1, 4, 1],
                "Valeurs": [
                    "CONSEIL",
                    "Société SAINT-GOBAIN",
                    "SAINT-GOBAIN",
                ],
                "Supprimer": [False, False, False],
                "Modifier": [False, False, False],
                "Fusionner": [False, False, False],
            }
        )

    with st.form("edit_form"):
        edited_df = st.data_editor(
            st.session_state.data,
            column_config={
                "Supprimer": st.column_config.CheckboxColumn("Supprimer"),
                "Modifier": st.column_config.CheckboxColumn("Modifier"),
                "Fusionner": st.column_config.CheckboxColumn("Fusionner"),
            },
            key="data_table",
        )
        submit = st.form_submit_button("Valider les sélections")

    if submit:
        st.session_state.data = edited_df
        rows_to_drop = st.session_state.data.index[st.session_state.data["Supprimer"]]
        if not rows_to_drop.empty:
            st.session_state.data.drop(rows_to_drop, inplace=True)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
