from __future__ import annotations
from typing import IO, Union
import pandas as pd

def read_csv_any(file_or_buffer: Union[str, IO[bytes], IO[str]], **kwargs) -> pd.DataFrame:
    """
    Read CSV from path/handle with graceful fallbacks for common delimiters/decimals.
    """
    try:
        return pd.read_csv(file_or_buffer, **kwargs)
    except Exception:
        # Try semicolon-delimited with comma decimals (EU-style)
        if hasattr(file_or_buffer, "seek"):
            file_or_buffer.seek(0)
        return pd.read_csv(file_or_buffer, sep=";", decimal=",", **kwargs)

# def read_csv_any(file_or_buffer, **kwargs):
#     """Read CSV from upload or bytes, with a few sensible defaults."""
#     try:
#         return pd.read_csv(file_or_buffer, **kwargs)
#     except Exception:
#         # Try semicolon delimiter and comma decimal as fallback
#         file_or_buffer.seek(0) if hasattr(file_or_buffer, "seek") else None
#         return pd.read_csv(file_or_buffer, sep=";", decimal=",", **kwargs)