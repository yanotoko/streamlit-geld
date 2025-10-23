# utils/io.py
from __future__ import annotations
from typing import IO, Union, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from io import BytesIO
import hashlib

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


# Transaction io
def _tx_norm_amount(x):
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return 0.0
    # Treat negative as expense; make positive magnitude
    return abs(float(v))

def _tx_make_id(row: pd.Series) -> str:
    base = f"{row.get('Date','')}|{row.get('Description','')}|{row.get('Merchant','')}|{row.get('Account','')}|{row.get('Amount','')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def read_transactions_file(file_like, mapping: dict) -> pd.DataFrame:
    """
    mapping keys: Date, Description, Merchant (opt), Account, Amount
    Returns normalized DF with TX_HEADERS + TX_ID & Source='upload'
    """
    raw = file_like.getvalue() if hasattr(file_like, "getvalue") else file_like.read()
    bio = BytesIO(raw)

    # Try Excel first, then CSV (fallbacks)
    try:
        df = pd.read_excel(bio)
    except Exception:
        bio.seek(0)
        try:
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, sep=";", decimal=",")

    # Map columns
    col_map = {
        "Date": mapping.get("Date"),
        "Description": mapping.get("Description"),
        "Merchant": mapping.get("Merchant"),  # may be None
        "Account": mapping.get("Account"),
        "Amount": mapping.get("Amount"),
    }
    missing = [k for k, src in col_map.items() if k != "Merchant" and (src is None or src not in df.columns)]
    if missing:
        raise ValueError(f"Missing required columns in upload: {', '.join(missing)}")

    out = pd.DataFrame()
    for k, src in col_map.items():
        if src and src in df.columns:
            out[k] = df[src]
        else:
            out[k] = ""

    # Normalize types
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    out["Description"] = out["Description"].astype(str).str.strip()
    out["Merchant"] = out["Merchant"].astype(str).str.strip()
    out["Account"] = out["Account"].astype(str).str.strip()
    out["Amount"] = out["Amount"].apply(_tx_norm_amount)

    # Defaults
    out["AssignedCategory"] = ""
    out["Source"] = "upload"
    out["TX_ID"] = out.apply(_tx_make_id, axis=1)

    # Keep canonical order
    cols = ["Date","Description","Merchant","Account","Amount","AssignedCategory","Source","TX_ID"]
    return out[cols]
