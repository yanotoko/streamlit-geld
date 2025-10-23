# utils/sheets.py
from __future__ import annotations
from typing import Tuple, Dict, List, Optional
import re
import hashlib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from utils.constants import LOOKUP_SHEET_PREFIX, LOOKUP_HEADERS, TX_SHEET_PREFIX, MERCHANTS_SHEET_PREFIX


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class VersionConflict(Exception):
    """Raised when the sheet has changed since the client loaded it."""


def _get_client() -> gspread.client.Client:
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)


def sanitize_workspace_id(raw: str) -> str:
    """
    Convert an arbitrary Workspace ID into a safe Google Sheets tab title.
    - lowercases, trims
    - spaces -> _
    - forbidden chars -> -
    - length cap (Sheets tab limit is 100; we keep margin)
    """
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[:\\/?*\[\]]", "-", s)
    s = re.sub(r"_+", "_", s)
    return s[:80] or "default"


def _hash_columns(cols: List[str]) -> str:
    h = hashlib.sha1("|".join(map(str, cols)).encode("utf-8")).hexdigest()
    return h[:12]


def _open_spreadsheet():
    client = _get_client()
    ss_id = st.secrets["sheets"]["spreadsheet_id"]
    return client.open_by_key(ss_id)


def _ensure_meta_sheet(spreadsheet) -> gspread.Worksheet:
    """Create a single _meta sheet to track versions per workspace."""
    try:
        ws = spreadsheet.worksheet("_meta")
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="_meta", rows=200, cols=6)
        set_with_dataframe(
            ws,
            pd.DataFrame(
                columns=[
                    "tab_title",
                    "workspace_id_original",
                    "version",
                    "updated_at",
                    "schema_hash",
                ]
            ),
        )
    return ws


def _get_meta_df(meta_ws: gspread.Worksheet) -> pd.DataFrame:
    df = get_as_dataframe(meta_ws, evaluate_formulas=True, header=0, dtype=str)
    if df is None or df.empty or "tab_title" not in df.columns:
        df = pd.DataFrame(
            columns=[
                "tab_title",
                "workspace_id_original",
                "version",
                "updated_at",
                "schema_hash",
            ]
        )
        set_with_dataframe(meta_ws, df)
    return df.fillna("")


def _upsert_meta(
    meta_ws: gspread.Worksheet,
    meta_df: pd.DataFrame,
    tab_title: str,
    workspace_id_original: str,
    version: int,
    schema_hash: str,
) -> Dict[str, str]:
    updated_at = datetime.now(timezone.utc).isoformat()
    mask = meta_df["tab_title"] == tab_title
    if not mask.any():
        new_row = {
            "tab_title": tab_title,
            "workspace_id_original": workspace_id_original,
            "version": int(version),
            "updated_at": updated_at,
            "schema_hash": schema_hash,
        }
        meta_df = pd.concat([meta_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = mask.idxmax()
        meta_df.at[idx, "workspace_id_original"] = workspace_id_original
        meta_df.at[idx, "version"] = int(version)
        meta_df.at[idx, "updated_at"] = updated_at
        meta_df.at[idx, "schema_hash"] = schema_hash
    set_with_dataframe(meta_ws, meta_df)
    return {
        "tab_title": tab_title,
        "workspace_id_original": workspace_id_original,
        "version": int(version),
        "updated_at": updated_at,
        "schema_hash": schema_hash,
    }


def _ensure_worksheet(spreadsheet, tab_title: str, headers: List[str]) -> gspread.Worksheet:
    """Ensure a worksheet exists, with headers in row 1."""
    try:
        ws = spreadsheet.worksheet(tab_title)
        # If the sheet exists but has no headers, set them.
        first_row = ws.row_values(1)
        if not first_row:
            ws.update("A1", [headers])
        return ws
    except gspread.WorksheetNotFound:
        rows = max(200, len(headers) + 20)
        cols = max(10, len(headers) + 2)
        ws = spreadsheet.add_worksheet(title=tab_title, rows=rows, cols=cols)
        ws.update("A1", [headers])
        return ws


def list_workspaces() -> List[str]:
    """Return known workspace IDs from the meta sheet (original IDs as entered by users)."""
    ss = _open_spreadsheet()
    meta_ws = _ensure_meta_sheet(ss)
    meta_df = _get_meta_df(meta_ws)
    if meta_df.empty:
        return []
    return (
        meta_df["workspace_id_original"]
        .dropna()
        .astype(str)
        .sort_values()
        .tolist()
    )


def read_workspace(workspace_id: str, default_headers: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a workspace. If its tab doesn't exist, create it with the provided headers.
    Returns (df, meta) and ensures a meta record exists.
    """
    tab_title = sanitize_workspace_id(workspace_id)
    ss = _open_spreadsheet()
    ws = _ensure_worksheet(ss, tab_title, headers=default_headers)

    df = get_as_dataframe(ws, evaluate_formulas=True, header=0, dtype=None)
    if df is None or df.empty:
        df = pd.DataFrame(columns=default_headers)
    else:
        # Drop completely empty rows (common when Sheets keeps extra rows)
        df = df.dropna(how="all")
        # Ensure all default headers exist if sheet was edited manually
        for col in default_headers:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        # Reorder columns to start with the defaults (any extras go to the end)
        default_first = [c for c in default_headers if c in df.columns]
        others = [c for c in df.columns if c not in default_first]
        df = df[default_first + others]

    meta_ws = _ensure_meta_sheet(ss)
    meta_df = _get_meta_df(meta_ws)
    rec = meta_df.loc[meta_df["tab_title"] == tab_title]
    if rec.empty:
        rec = _upsert_meta(
            meta_ws=meta_ws,
            meta_df=meta_df,
            tab_title=tab_title,
            workspace_id_original=workspace_id,
            version=0,
            schema_hash=_hash_columns(list(df.columns)),
        )
    else:
        rec = rec.iloc[0].to_dict()
        rec["version"] = int(rec.get("version", 0))

    rec.update({"tab_title": tab_title, "workspace_id": workspace_id})
    return df, rec


def write_workspace(
    workspace_id: str, df: pd.DataFrame, expected_version: Optional[int]
) -> int:
    """
    Save the DataFrame to the user's worksheet with optimistic locking.
    Returns the new version on success.
    """
    tab_title = sanitize_workspace_id(workspace_id)
    ss = _open_spreadsheet()
    ws = _ensure_worksheet(ss, tab_title, headers=list(df.columns))

    # Check version
    meta_ws = _ensure_meta_sheet(ss)
    meta_df = _get_meta_df(meta_ws)
    rec = meta_df.loc[meta_df["tab_title"] == tab_title]
    current_version = int(rec.iloc[0]["version"]) if not rec.empty else 0

    if expected_version is not None and int(expected_version) != current_version:
        raise VersionConflict(
            f"Workspace changed: your version={expected_version}, latest={current_version}"
        )

    # Write (clear + full data write with headers)
    ws.clear()
    set_with_dataframe(ws,
                        df,
                        include_index=False,             # <- do not write the index
                        include_column_header=True,
                        resize=True
)

    # Bump version & update meta
    new_version = current_version + 1
    _upsert_meta(
        meta_ws=meta_ws,
        meta_df=meta_df,
        tab_title=tab_title,
        workspace_id_original=workspace_id,
        version=new_version,
        schema_hash=_hash_columns(list(df.columns)),
    )
    return new_version

# --- Per-workspace lookup sheet helpers ---

def _lookup_sheet_title_for_ws(workspace_id: str) -> str:
    """
    Build the lookup worksheet title for a given workspace.
    Example: IncomeMaster__team_alpha
    """
    tab_title = sanitize_workspace_id(workspace_id)
    return f"{LOOKUP_SHEET_PREFIX}__{tab_title}" if tab_title else LOOKUP_SHEET_PREFIX


def ensure_lookup_sheet_for_ws(spreadsheet, workspace_id: str, headers: List[str], starter_rows: Optional[List[Dict]] = None) -> gspread.Worksheet:
    """
    Ensure a per-workspace lookup sheet exists with the given headers.
    If starter_rows is provided and the sheet is empty, seed it.
    """
    title = _lookup_sheet_title_for_ws(workspace_id)
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=200, cols=max(10, len(headers) + 2))
        ws.update("A1", [headers])
        if starter_rows:
            df = pd.DataFrame(starter_rows, columns=headers)
            set_with_dataframe(ws, df, include_index=False, include_column_header=True)
        return ws

    # If existing but empty (no headers), set headers and seed
    first_row = ws.row_values(1)
    if not first_row:
        ws.update("A1", [headers])
        if starter_rows:
            df = pd.DataFrame(starter_rows, columns=headers)
            set_with_dataframe(ws, df, include_index=False, include_column_header=True)
    return ws


def read_lookup_for_ws(workspace_id: str, starter_rows: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Read the per-workspace lookup sheet. If it doesn't exist, create it.
    If it's empty, optionally seed with starter_rows.
    """
    ss = _open_spreadsheet()
    ws = ensure_lookup_sheet_for_ws(ss, workspace_id, LOOKUP_HEADERS, starter_rows)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0, dtype=None)
    if df is None or df.empty:
        return pd.DataFrame(columns=LOOKUP_HEADERS)

    # Keep only known columns, normalize types
    df = df[LOOKUP_HEADERS].copy()
    df["Level1"] = df["Level1"].astype(str).str.strip()
    df["Frequency"] = df["Frequency"].astype(str).str.strip()
    df["Factor_per_month"] = pd.to_numeric(df["Factor_per_month"], errors="coerce").fillna(1.0)
    return df


def write_lookup_for_ws(workspace_id: str, df: pd.DataFrame) -> None:
    """
    Overwrite the per-workspace lookup sheet with df.
    """
    ss = _open_spreadsheet()
    ws = ensure_lookup_sheet_for_ws(ss, workspace_id, LOOKUP_HEADERS)
    ws.clear()
    out = pd.DataFrame(df, columns=LOOKUP_HEADERS)
    set_with_dataframe(ws, out, include_index=False, include_column_header=True, resize=True)

# ---------- Category Universe (per workspace) ----------
META_HEADERS = ["Key", "Value"]

def _meta_sheet_title_for_ws(workspace_id: str) -> str:
    tab = sanitize_workspace_id(workspace_id)
    return f"Meta__{tab}" if tab else "Meta__default"

def _ensure_meta_for_ws(spreadsheet, workspace_id: str):
    title = _meta_sheet_title_for_ws(workspace_id)
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=50, cols=4)
        ws.update("A1", [META_HEADERS])
    # seed headers if empty
    if not ws.row_values(1):
        ws.update("A1", [META_HEADERS])
    return ws

def read_category_universe(workspace_id: str) -> str | None:
    """Return the saved workbook column name used as the Category Universe, or None if not set."""
    ss = _open_spreadsheet()
    ws = _ensure_meta_for_ws(ss, workspace_id)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0, dtype=str)
    if df is None or df.empty:
        return None
    df = df[META_HEADERS].dropna(how="all").fillna("")
    rec = df.loc[df["Key"] == "CategoryUniverse"]
    if rec.empty:
        return None
    val = rec.iloc[0]["Value"].strip()
    return val or None

def write_category_universe(workspace_id: str, column_name: str) -> None:
    """Persist the chosen workbook column as the Category Universe for this workspace."""
    ss = _open_spreadsheet()
    ws = _ensure_meta_for_ws(ss, workspace_id)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0, dtype=str)
    if df is None or df.empty:
        df = pd.DataFrame(columns=META_HEADERS)
    df = df[META_HEADERS].fillna("")
    mask = df["Key"] == "CategoryUniverse"
    if mask.any():
        idx = mask.idxmax()
        df.at[idx, "Value"] = column_name
    else:
        df = pd.concat([df, pd.DataFrame([{"Key": "CategoryUniverse", "Value": column_name}])], ignore_index=True)
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)


# ---------- Transactions Helpers----------
def _tx_sheet_title_for_ws(workspace_id: str) -> str:
    tab = sanitize_workspace_id(workspace_id)
    return f"{TX_SHEET_PREFIX}__{tab}"

def _merchants_sheet_title_for_ws(workspace_id: str) -> str:
    tab = sanitize_workspace_id(workspace_id)
    return f"{MERCHANTS_SHEET_PREFIX}__{tab}"

def _ensure_ws_with_headers(spreadsheet, title: str, headers: list[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=400, cols=max(10, len(headers)+2))
        ws.update("A1", [headers])
        return ws
    # seed headers if empty
    if not ws.row_values(1):
        ws.update("A1", [headers])
    return ws

# ---------- Transactions ----------
TX_HEADERS = ["Date","Description","Merchant","Account","Amount","AssignedCategory","Source","TX_ID"]

def read_transactions(workspace_id: str) -> pd.DataFrame:
    """
    Read the Transactions sheet for a workspace, preserving any extra columns that
    may exist in the sheet. Ensure expected TX_HEADERS exist; return with expected
    columns first, then any extras.
    """
    ss = _open_spreadsheet()
    ws = _ensure_ws_with_headers(ss, _tx_sheet_title_for_ws(workspace_id), TX_HEADERS)

    values = ws.get_all_values()
    if not values:
        # brand-new sheet
        return pd.DataFrame(columns=TX_HEADERS)

    header = [h.strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Ensure all expected columns exist (don’t drop extras)
    for c in TX_HEADERS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

    # Order: expected first, then extras (preserve sheet order for extras)
    extras = [c for c in df.columns if c not in TX_HEADERS]
    ordered_cols = TX_HEADERS + extras

    # Drop all-empty rows (if any)
    df = df[ordered_cols].dropna(how="all")
    return df

def write_transactions(workspace_id: str, df: pd.DataFrame) -> None:
    """
    Schema-flexible write: union existing sheet columns with incoming df columns,
    keep existing order first, then add any new columns at the end. Clears sheet
    and writes header + data accordingly.
    """
    ss = _open_spreadsheet()
    ws = _ensure_ws_with_headers(ss, _tx_sheet_title_for_ws(workspace_id), TX_HEADERS)

    # Current header (if any)
    existing_values = ws.get_all_values()
    existing_header = [h.strip() for h in existing_values[0]] if existing_values else []

    # Normalize incoming columns to strings
    new_cols = [str(c).strip() for c in df.columns.tolist()]

    # Ensure all TX_HEADERS are present in the union too (future-proofing)
    baseline = existing_header if existing_header else TX_HEADERS
    union_cols = baseline + [c for c in new_cols if c not in baseline]
    # If baseline was TX_HEADERS and df has extras, they’ll be appended

    # Reindex to union; convert NaN to "" for Sheets
    out = df.copy()
    out.columns = new_cols  # apply normalized names
    out = out.reindex(columns=union_cols)
    out = out.astype(object).where(pd.notnull(out), "")

    # Clear then write header + rows
    ws.clear()
    set_with_dataframe(
        ws,
        out,
        include_index=False,
        include_column_header=True,
        resize=True
    )

# ---------- Merchants memory ----------
MERCHANT_HEADERS = ["MerchantKey","AssignedCategory"]

def read_merchants(workspace_id: str) -> pd.DataFrame:
    ss = _open_spreadsheet()
    ws = _ensure_ws_with_headers(ss, _merchants_sheet_title_for_ws(workspace_id), MERCHANT_HEADERS)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0, dtype=None)
    if df is None or df.empty:
        return pd.DataFrame(columns=MERCHANT_HEADERS)
    for c in MERCHANT_HEADERS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    df["MerchantKey"] = df["MerchantKey"].astype(str).str.strip().str.lower()
    df["AssignedCategory"] = df["AssignedCategory"].astype(str).str.strip()
    return df[MERCHANT_HEADERS].dropna(how="all")

def write_merchants(workspace_id: str, df: pd.DataFrame) -> None:
    ss = _open_spreadsheet()
    ws = _ensure_ws_with_headers(ss, _merchants_sheet_title_for_ws(workspace_id), MERCHANT_HEADERS)
    out = pd.DataFrame(df, columns=MERCHANT_HEADERS)
    ws.clear()
    set_with_dataframe(ws, out, include_index=False, include_column_header=True, resize=True)
