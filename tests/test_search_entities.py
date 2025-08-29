import pages.recherche_avancee as ra
import streamlit as st


def test_search_entities_groups_occurrences():
    text = "Paris est belle. Paris est la capitale de la France."
    results = ra.search_entities(text, r"Paris", context_window=5)
    assert "Paris" in results
    info = results["Paris"]
    assert info["nb_occurrences"] == 2
    # Ensure we captured context for each occurrence
    assert len(info["contextes"]) == 2
    assert all("Paris" in ctx for ctx in info["contextes"])


def test_add_entity_updates_group_store():
    st.session_state.clear()
    info = {"nb_occurrences": 1, "contextes": ["Paris"]}
    store: list = []
    ra.add_entity("Paris", "ORG_99", info, store)
    assert store[0]["valeur_anonymisation"] == "ORG_99"
    assert "data" in st.session_state
    df = st.session_state.data
    assert df.iloc[0]["Token"] == "ORG_99"
    assert df.iloc[0]["Valeurs"] == "Paris"
