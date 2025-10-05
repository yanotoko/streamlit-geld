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
    set_with_dataframe(ws, df)

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
