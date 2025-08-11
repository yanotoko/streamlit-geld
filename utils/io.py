# utils/io.py
from __future__ import annotations
from typing import IO, Union, List, Optional, Tuple
from pathlib import Path
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

def get_excel_sheets(file_obj) -> List[str]:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    xls = pd.ExcelFile(file_obj)
    sheets = xls.sheet_names
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return sheets

def read_table_any(file_obj, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> pd.DataFrame:
    name = getattr(file_obj, "name", None)
    ext = Path(name or "").suffix.lower()
    is_excel = ext in {".xlsx", ".xls", ".xlsm"}
    if is_excel:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return pd.read_excel(file_obj, sheet_name=sheet_name, **kwargs)
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return read_csv_any(file_obj, **kwargs)

def validate_upload(
    uploaded,
    allowed_exts: set,
    allowed_mime: set,
    max_bytes: int
) -> Tuple[bool, str]:
    """Basic preflight checks for Streamlit UploadedFile."""
    if uploaded is None:
        return False, "No file uploaded."
    name = getattr(uploaded, "name", "") or ""
    ext = Path(name).suffix.lower()
    size = getattr(uploaded, "size", None)
    mime = getattr(uploaded, "type", "") or ""

    errs = []
    if ext not in allowed_exts:
        errs.append(f"Unsupported extension '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}.")
    # Be tolerant of 'application/octet-stream' but still require a good extension
    if mime not in allowed_mime:
        errs.append(f"Unexpected MIME type '{mime}'.")
    if isinstance(size, int) and size > max_bytes:
        errs.append(f"File too large ({size/1_000_000:.1f} MB). Max {max_bytes/1_000_000:.1f} MB.")

    return (len(errs) == 0), " ".join(errs)
