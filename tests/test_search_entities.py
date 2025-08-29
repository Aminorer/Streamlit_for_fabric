import pages.recherche_avancee as ra


def test_search_entities_groups_occurrences():
    text = "Paris est belle. Paris est la capitale de la France."
    results = ra.search_entities(text, r"Paris", context_window=5)
    assert "Paris" in results
    info = results["Paris"]
    assert info["nb_occurrences"] == 2
    # Ensure we captured context for each occurrence
    assert len(info["contextes"]) == 2
    assert all("Paris" in ctx for ctx in info["contextes"]) 
