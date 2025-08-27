import re
from typing import Iterable, List

def sanitize_input(value: str, pattern: str = r'^[\w\s-]{1,100}$') -> str:
    """Validate and clean a single user input.

    Parameters
    ----------
    value : str
        Input string to validate.
    pattern : str, optional
        Regular expression describing the expected format.
        Defaults to ``r'^[\\w\\s-]{1,100}$'`` allowing letters, digits,
        spaces, underscores and hyphens (1 to 100 characters).

    Returns
    -------
    str
        The original string if it is valid.

    Raises
    ------
    ValueError
        If the input is empty or does not match the provided pattern.
    """
    if not isinstance(value, str) or not value or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid input: {value!r}")
    return value.strip()


def sanitize_list(values: Iterable[str], pattern: str = r'^[\w\s-]{1,100}$') -> List[str]:
    """Validate each element of an iterable using :func:`sanitize_input`.

    Parameters
    ----------
    values : Iterable[str]
        Iterable of strings to validate.
    pattern : str, optional
        Regular expression to use for validation. Same default as
        :func:`sanitize_input`.

    Returns
    -------
    List[str]
        List of validated strings.

    Raises
    ------
    ValueError
        If any element fails validation.
    """
    return [sanitize_input(v, pattern) for v in values]

