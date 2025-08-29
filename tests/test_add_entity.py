import pytest
import pages.recherche_avancee as ra


def test_add_entity_stores_entity():
    store = []
    info = {"nb_occurrences": 1, "contextes": ["ctx"]}
    ra.add_entity("Paris", "FR", info, store)
    assert store == [
        {
            "token": "Paris",
            "valeur_anonymisation": "FR",
            "nb_occurrences": 1,
            "contextes": ["ctx"],
        }
    ]


def test_add_entity_requires_value():
    store = []
    info = {"nb_occurrences": 1, "contextes": []}
    with pytest.raises(ValueError):
        ra.add_entity("Paris", "", info, store)
