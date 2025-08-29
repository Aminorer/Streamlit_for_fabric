import re
from typing import Dict, List

import streamlit as st


def search_entities(text: str, pattern: str, context_window: int = 40) -> Dict[str, Dict[str, List[str]]]:
    """Search ``text`` for ``pattern`` and group identical occurrences.

    Parameters
    ----------
    text: str
        The text to search.
    pattern: str
        A regular expression pattern describing the entity to search for.
    context_window: int, optional
        Number of characters to include before and after each match for context.

    Returns
    -------
    dict
        Mapping of the matched string to a dictionary with:
        ``nb_occurrences`` – count of matches,
        ``contextes`` – list of surrounding text for each match.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    results: Dict[str, Dict[str, List[str]]] = {}
    for match in regex.finditer(text):
        token = match.group(0)
        start, end = match.span()
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end]
        if token not in results:
            results[token] = {"nb_occurrences": 0, "contextes": []}
        results[token]["nb_occurrences"] += 1
        results[token]["contextes"].append(context)
    return results


def main() -> None:
    st.set_page_config(page_title="Recherche avancée")
    st.title("Recherche avancée")

    st.write(
        """Saisissez un texte puis une expression régulière pour rechercher des entités.\n"
        "Les occurrences identiques sont regroupées et leur nombre est indiqué."
        """
    )
    text = st.text_area("Texte à analyser")
    pattern = st.text_input("Expression à rechercher")

    if pattern:
        results = search_entities(text, pattern)
        if not results:
            st.info("Aucune occurrence trouvée.")
        for token, info in results.items():
            st.subheader(f"{token} — {info['nb_occurrences']} occurrence(s)")
            with st.expander("Voir les contextes"):
                for idx, ctx in enumerate(info["contextes"], start=1):
                    st.write(f"{idx}. …{ctx}…")


if __name__ == "__main__":
    main()
