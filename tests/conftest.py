import logging
import pytest

@pytest.fixture(autouse=True)
def silence_streamlit_cache_warnings():
    logging.getLogger(
        "streamlit.runtime.caching.cache_data_api"
    ).setLevel(logging.ERROR)
