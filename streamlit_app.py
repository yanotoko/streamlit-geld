# app/streamlit_app.py
import io
import pandas as pd
import re
import hashlib
import streamlit as st
from streamlit_echarts import st_echarts

from utils.echarts import make_echarts_sankey_options
from utils.io import read_csv_any, read_table_any, get_excel_sheets, validate_upload
from utils.sankey import prepare_sankey_data, make_sankey_figure
from utils.default_catalog import DEFAULT_INCOME_SOURCES, flat_default_categories, defaults_for
from utils.constants import SAMPLE_CSV, DEFAULT_HEADERS, ALLOWED_EXTS, ALLOWED_MIME, MAX_UPLOAD_MB, LOOKUP_HEADERS, FREQUENCY_CHOICES
from utils.sheets import read_workspace, write_workspace, sanitize_workspace_id, VersionConflict, read_lookup_for_ws, write_lookup_for_ws

from utils.io import read_transactions_file  # NEW
from utils.sheets import read_transactions, write_transactions, read_merchants, write_merchants, read_category_universe, write_category_universe

from rapidfuzz import process, fuzz  # NEW

# =======================================================================================
# Helper functions
# =======================================================================================

# ---- Snapshots helper so every tab can safely read the same 3 frames
def get_snapshots(df_raw_fallback: pd.DataFrame):
    budget = st.session_state.get("ws_budget_df", df_raw_fallback.copy())
    tx     = st.session_state.get("ws_tx_df", st.session_state.get("tx_staged", pd.DataFrame()))
    lookup = st.session_state.get("ws_lookup_df", pd.DataFrame(columns=LOOKUP_HEADERS))
    return budget, tx, lookup


# ============= Guided Builder helpers =============

def _detect_subcategory_column(df: pd.DataFrame) -> str | None:
    """Try to find an existing Subcategory column."""
    if df is None or df.empty:
        return None
    for cand in ["Subcategory", "Sub-category", "Sub Category"]:
        if cand in df.columns:
            return cand
    # fallback: none
    return None

def _ensure_subcategory_column(df: pd.DataFrame, subcat_col_name: str = "Subcategory") -> pd.DataFrame:
    """Add a Subcategory column if missing (keeps existing data intact)."""
    if subcat_col_name not in df.columns:
        df = df.copy()
        df[subcat_col_name] = ""
    return df

def _load_l1_factors_from_lookup(lookup_df: pd.DataFrame) -> dict[str, float]:
    """Convert lookup table to a dict Level1 -> Factor_per_month."""
    if lookup_df is None or lookup_df.empty:
        return {}
    out = {}
    for _, r in lookup_df.iterrows():
        l1v = str(r.get("Level1", "")).strip()
        fv  = pd.to_numeric(r.get("Factor_per_month", 1.0), errors="coerce")
        if l1v:
            out[l1v] = float(fv if pd.notna(fv) else 1.0)
    return out

def _write_l1_factor(ws_id: str, lookup_df: pd.DataFrame, level1_val: str, factor: float) -> pd.DataFrame:
    """Upsert (Level1 -> Factor_per_month) in lookup, persist to Sheets & return fresh df."""
    # Normalize
    level1_val = str(level1_val).strip()
    factor = float(factor if factor is not None else 1.0)

    # Prepare updated lookup dataframe
    base = lookup_df.copy() if lookup_df is not None else pd.DataFrame(columns=["Level1", "Frequency", "Factor_per_month"])
    if "Level1" not in base.columns: base["Level1"] = ""
    if "Frequency" not in base.columns: base["Frequency"] = ""
    if "Factor_per_month" not in base.columns: base["Factor_per_month"] = 1.0

    mask = base["Level1"].astype(str).str.strip().eq(level1_val)
    if mask.any():
        base.loc[mask, "Factor_per_month"] = factor
        if "Frequency" in base.columns and base.loc[mask, "Frequency"].isna().any():
            base.loc[mask, "Frequency"] = "monthly"
    else:
        base = pd.concat([base, pd.DataFrame([{
            "Level1": level1_val,
            "Frequency": "monthly",
            "Factor_per_month": factor
        }])], ignore_index=True)

    # Persist if workspace is available
    if ws_id:
        try:
            write_lookup_for_ws(ws_id, base)
        except Exception as e:
            st.warning(f"Could not save factor for '{level1_val}': {e}")

    return base

def _account_col_from_budget(df: pd.DataFrame) -> str | None:
    """Find the budget-side account column (Bank Account / Account / Destination)."""
    if df is None or df.empty:
        return None
    priority = ["Bank Account", "Account", "Destination"]
    cols = list(df.columns)
    # exact case-insensitive
    for target in priority:
        for c in cols:
            if str(c).strip().lower() == target.lower():
                return c
    # substring fallback
    for target in priority:
        for c in cols:
            if target.lower() in str(c).strip().lower():
                return c
    return None

# ---- Fuzzy suggestion helper (RapidFuzz)
def _suggest_category_universe_with_score(text: str, universe: list[str], cutoff: int = 10) -> tuple[str, int]:
    if not text or not universe:
        return ("", 0)
    match = process.extractOne(text, universe, scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
    if not match:
        return ("", 0)
    label, score = match[0], int(match[1])
    return (label, score)

def _to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _ensure_tx_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a stable TX_ID column exists. Uses Date|Description|Merchant|Account|Amount."""
    needed = ["Date", "Description", "Merchant", "Account", "Amount"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""  # safe default; better than erroring

    key_series = (
        df["Date"].astype(str).str.strip() + "|" +
        df["Description"].astype(str).str.strip() + "|" +
        df["Merchant"].astype(str).str.strip() + "|" +
        df["Account"].astype(str).str.strip() + "|" +
        df["Amount"].astype(str).str.strip()
    )
    df["TX_ID"] = key_series.apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    return df

def _norm_merchant_key(s: str) -> str:
    """Normalize merchant names for memory keys (must match lookup logic)."""
    return (str(s).lower().replace("/", " ").strip() if s is not None else "")

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

@st.cache_data(ttl=20, show_spinner=False)
def _cached_read_workspace(ws_id: str, default_headers, version_key: int, cache_bump: int):
    return read_workspace(ws_id, default_headers)

@st.cache_data(ttl=20, show_spinner=False)
def _cached_read_lookup(ws_id: str, cache_bump: int):
    return read_lookup_for_ws(ws_id, starter_rows=None)

def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

st.set_page_config(page_title="Budget App", layout="wide")
st.title("💰 Budget App")
# =============== UI Helpers (icons, headers, spacing) + Theme ===============
THEME_BASE = (st.get_option("theme.base") or "light").lower()
IS_DARK = THEME_BASE == "dark"
PLOTLY_TEMPLATE = "plotly_dark" if IS_DARK else "plotly_white"
ECHR_THEME = "dark" if IS_DARK else "light"

# CSS that reads Streamlit theme variables (works in light & dark)
st.markdown("""
<style>
:root {
  --hdr-color: var(--text-color);
  --cap-color: var(--secondary-text-color);
  --chip-bg: var(--secondary-background-color);
  --chip-fg: var(--primary-color);
  --chip-border: var(--primary-color);
}

.h2-line {
  display:flex; align-items:center; gap:.6rem;
  font-weight:700; font-size:1.25rem; color: var(--hdr-color);
  margin: 8px 0 2px 0;
}
.h2-cap {
  font-size:.95rem; color: var(--cap-color); margin: 0 0 6px 0;
}
.sb-h {
  font-weight:700; font-size:1.0rem; color: var(--hdr-color);
  margin: 4px 0 8px 0; display:flex; gap:.5rem; align-items:center;
}
.tag {
  display:inline-block; padding:.25rem .55rem; border-radius:999px;
  background: var(--chip-bg);
  color: var(--chip-fg);
  border: 1px solid var(--chip-border);
  font-weight:600; font-size:.85rem;
}
</style>
""", unsafe_allow_html=True)

def section_header(icon: str, title: str, caption: str | None = None):
    st.divider()
    st.markdown(f'<div class="h2-line">{icon} {title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="h2-cap">{caption}</div>', unsafe_allow_html=True)

def sidebar_header(icon: str, title: str):
    st.sidebar.markdown(f'<div class="sb-h">{icon} {title}</div>', unsafe_allow_html=True)

def badge(text: str):
    st.markdown(f'<span class="tag">{text}</span>', unsafe_allow_html=True)


# =======================================================================================
# SIDEBAR (workspace, transactions upload, data source)
# =======================================================================================

# 1) WORKSPACE
sidebar_header("🧭", "1) Workspace")
with st.sidebar.form(key="ws_form", clear_on_submit=False):
    ws_input = st.text_input(
        "Workspace ID",
        value=st.session_state.get("workspace_id", ""),
        help="Public app: any non-PII name. Data will be saved under this ID."
    )
    ws_submit = st.form_submit_button("Use workspace")
if ws_submit:
    st.session_state["workspace_id"] = ws_input.strip()

ws_id = st.session_state.get("workspace_id", "").strip()
ws_tab = sanitize_workspace_id(ws_id) if ws_id else ""
st.session_state.setdefault("cache_bump", 0)

# 2) TRANSACTIONS UPLOAD (moved here, right after Workspace)
st.sidebar.divider()
sidebar_header("🧾", "2) Transactions upload")

tx_file_sb = st.sidebar.file_uploader(
    "Upload transactions (CSV/XLSX)",
    type=["csv","xlsx","xls","xlsm"],
    key="tx_uploader_sidebar"
)
if tx_file_sb is not None:
    st.session_state["tx_file_latest"] = tx_file_sb

# 3) DATA SOURCE (workbook)
st.sidebar.divider()
sidebar_header("📒", "3) Budget workbook")
st.sidebar.caption("Load from a workspace or upload a file. You can import an upload into your workspace.")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=[ext.lstrip(".") for ext in ALLOWED_EXTS], key="budget_upload")

# Parse uploaded (if any)
df_uploaded, upload_name = None, None
if uploaded is not None:
    ok, msg = validate_upload(
        uploaded,
        allowed_exts=ALLOWED_EXTS,
        allowed_mime=ALLOWED_MIME,
        max_bytes=MAX_UPLOAD_MB * 1024 * 1024,
    )
    if not ok:
        st.error(msg)
        st.stop()

    upload_name = uploaded.name
    ext = upload_name.split(".")[-1].lower()
    if ext in {"xlsx", "xls", "xlsm"}:
        sheets = get_excel_sheets(uploaded)
        sheet = st.sidebar.selectbox("Excel sheet", sheets, index=0)
        df_uploaded = read_table_any(uploaded, sheet_name=sheet)
    else:
        df_uploaded = read_table_any(uploaded)
    st.sidebar.caption(f"Detected file: **{upload_name}**")

# Try loading workspace (cached)
df_ws, ws_loaded, ws_empty = None, False, True
if ws_id:
    try:
        ws_ver = st.session_state.get("ws_version", 0)
        df_ws, meta = _cached_read_workspace(ws_id, DEFAULT_HEADERS, ws_ver, st.session_state["cache_bump"])
        ws_loaded, ws_empty = True, df_ws.dropna(how="all").shape[0] == 0
        st.session_state["ws_version"] = int(meta.get("version", 0))

        # right after reading df_ws/meta
        incoming = df_ws.copy()
        current = st.session_state.get("ws_budget_df")
        switched = st.session_state.get("_loaded_workspace_id") != ws_id
        incoming_has_rows = not incoming.empty
        current_empty = current is None or current.empty

        if switched or incoming_has_rows or current_empty:
            st.session_state["ws_budget_df"] = incoming

        st.session_state["_loaded_workspace_id"] = ws_id
        # ---------- NEW: snapshot everything for Overview ----------
        st.session_state["ws_budget_df"] = df_ws.copy()

        try:
            st.session_state["ws_tx_df"] = read_transactions(ws_id).copy()
        except Exception:
            st.session_state["ws_tx_df"] = pd.DataFrame()

        try:
            st.session_state["ws_lookup_df"] = _cached_read_lookup(ws_id, st.session_state["cache_bump"]).copy()
        except Exception:
            st.session_state["ws_lookup_df"] = pd.DataFrame(columns=LOOKUP_HEADERS)

        try:
            st.session_state["cu_col_saved"] = read_category_universe(ws_id)
        except Exception:
            st.session_state["cu_col_saved"] = None

        cu_col_ws = st.session_state.get("cu_col_saved") or st.session_state.get("cat_universe_col")
        if cu_col_ws and cu_col_ws in st.session_state["ws_budget_df"].columns:
            st.session_state["tx_universe"] = sorted(
                st.session_state["ws_budget_df"][cu_col_ws].dropna().astype(str).str.strip().unique().tolist()
            )
        # -----------------------------------------------------------
        # --- trigger a single rerun so Overview sees fresh snapshots immediately ---
        if not st.session_state.get("__ws_loaded_once"):
            st.session_state["__ws_loaded_once"] = True
            _rerun()

    except Exception as e:
        st.error(f"Could not load workspace '{ws_id}': {e}")



# Choose source
if ws_loaded and not ws_empty and df_uploaded is not None:
    source_choice = st.sidebar.radio("Use data from", ("Workspace", "Uploaded file"), index=0, horizontal=True)
elif ws_loaded and not ws_empty:
    source_choice = "Workspace"
elif df_uploaded is not None:
    source_choice = "Uploaded file"
else:
    source_choice = None

# Import uploaded → workspace
if ws_id and df_uploaded is not None:
    with st.sidebar.expander("Import uploaded into workspace", expanded=False):
        st.write("This will create or overwrite the workspace tab with the uploaded file.")
        col_imp1, col_imp2 = st.columns(2)
        if col_imp1.button("Import uploaded → Workspace", use_container_width=True):
            try:
                expected = None if ws_empty else st.session_state.get("ws_version", 0)
                new_version = write_workspace(ws_id, df_uploaded, expected_version=expected)
                st.session_state["ws_version"] = new_version
                st.success(f"Imported into '{ws_id}' (version {new_version}).") 
                st.session_state["cache_bump"] += 1
                _rerun()
            except VersionConflict as vc:
                st.warning(str(vc))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Reload workspace"):
                        _rerun()
                with c2:
                    if st.button("Force overwrite"):
                        try:
                            new_version = write_workspace(ws_id, df_uploaded, expected_version=None)
                            st.session_state["ws_version"] = new_version
                            st.success(f"Overwrote '{ws_id}' (version {new_version}).")
                            st.session_state["cache_bump"] += 1
                            _rerun()
                        except Exception as e:
                            st.error(f"Overwrite failed: {e}")

# Fallback sample downloader
st.sidebar.download_button(
    "Download sample.csv",
    data=SAMPLE_CSV,
    file_name="sample.csv",
    mime="text/csv",
    use_container_width=True
)

# Materialize df_raw + source_name
if source_choice == "Workspace":
    df_raw = df_ws.copy()
    source_name = f"Google Sheets (tab: {ws_tab})"
elif source_choice == "Uploaded file":
    df_raw = df_uploaded.copy()
    source_name = upload_name
else:
    st.info("No workspace data or upload found. Using the built-in **sample dataset**.")
    df_raw = read_csv_any(io.StringIO(SAMPLE_CSV))
    source_name = "sample.csv"

# ---------- NEW: ensure snapshots exist even before visiting Setup ----------
st.session_state.setdefault("ws_budget_df", df_raw.copy())
st.session_state.setdefault("ws_tx_df", st.session_state.get("tx_staged", pd.DataFrame()))
st.session_state.setdefault("ws_lookup_df", pd.DataFrame(columns=LOOKUP_HEADERS))
# Guide toggle default
st.session_state.setdefault("guide_active", False)
st.session_state.setdefault("__editor_rev", 0)

# ---------------------------------------------------------------------------

# 4) MAP COLUMNS
st.sidebar.divider()
sidebar_header("🗺️", "4) Map columns")
all_cols = list(df_raw.columns)

def guess(col_names, candidates):
    for c in candidates:
        if c in col_names:
            return c
    return col_names[0] if col_names else None

l1 = st.sidebar.selectbox("Level 1", all_cols, index=all_cols.index(guess(all_cols, ["Income"])) if all_cols else 0,
                          help="Top level (e.g., Income)")
l2 = st.sidebar.selectbox("Level 2", all_cols, index=all_cols.index(guess(all_cols, ["Bank Account","Account","Destination"])) if all_cols else 0,
                          help="Second level (e.g., Category)")
l3 = st.sidebar.selectbox("Level 3", all_cols, index=all_cols.index(guess(all_cols, ["Subcategory","Category"])) if all_cols else 0,
                          help="Third level (e.g., Bank Account)")

# Optional Level 4
none_l4_label = "— None —"
l4_options = [none_l4_label] + all_cols
l4_choice = st.sidebar.selectbox("Level 4 (optional)", l4_options, index=0,
    help="Fourth level (optional). Leave as None to use 3 levels.")
l4 = None if l4_choice == none_l4_label else l4_choice

val_col = st.sidebar.selectbox("Value (numeric)", all_cols, index=all_cols.index(guess(all_cols, ["Amount","Value","Count"])) if all_cols else 0)

chosen_levels = [l1, l2, l3] + ([l4] if l4 else [])
if len(set(chosen_levels)) < len(chosen_levels):
    st.sidebar.warning("Each selected level should be a different column.")

# Category Universe (persisted; independent from L2)
cat_universe_saved = None
if ws_id:
    try:
        cat_universe_saved = read_category_universe(ws_id)
    except Exception:
        cat_universe_saved = None

proposed_default = cat_universe_saved if (cat_universe_saved in all_cols) else (
    l2 if (l2 in all_cols) else (all_cols[0] if all_cols else "")
)

cat_universe_col = st.sidebar.selectbox(
    "Category universe column (from workbook)",
    options=all_cols,
    index=all_cols.index(proposed_default) if proposed_default in all_cols else 0,
    help="Transactions will be classified only into the distinct values of this workbook column."
)

cu_cols = st.sidebar.columns([1,1])
with cu_cols[0]:
    if st.button("Save category universe", use_container_width=True, disabled=not bool(ws_id)):
        try:
            write_category_universe(ws_id, cat_universe_col)
            st.session_state["cat_universe_col"] = cat_universe_col
            st.success(f"Saved Category Universe: {cat_universe_col}")
        except Exception as e:
            st.error(f"Could not save Category Universe: {e}")
with cu_cols[1]:
    if ws_id and st.button("Use saved", use_container_width=True, disabled=not bool(cat_universe_saved)):
        if cat_universe_saved:
            st.session_state["cat_universe_col"] = cat_universe_saved
        else:
            st.info("No saved Category Universe found.")

# 4) FILTER & VISUALIZATION
st.sidebar.divider()
sidebar_header("🪢", "Sankey chart setup")
with st.sidebar.expander("Sankey Chart Options", expanded=False):
    chart_type = st.selectbox("Chart type", ["Sankey (Plotly)", "Sankey (ECharts, labels always on)"], index=1)
    if chart_type.startswith("Sankey (ECharts"):
        ech_curveness = st.slider("Link curveness", 0.0, 1.0, 0.5, 0.05)
        ech_edge_font = st.number_input("Edge label font size", 8, 24, 11, 1)

st.sidebar.divider()

# --- Load per-workspace lookup (seed from current Level1 values if empty) ---
df_lookup = pd.DataFrame(columns=LOOKUP_HEADERS)  # safe default
if ws_id:
    starter = None
    if l1 in df_raw.columns:
        starter_levels = sorted(df_raw[l1].dropna().astype(str).str.strip().unique().tolist())
        if starter_levels:
            starter = [{"Level1": x, "Frequency": "monthly", "Factor_per_month": 1.0} for x in starter_levels]
    try:
        if "df_lookup_seeded" not in st.session_state and starter:
            try:
                _ = read_lookup_for_ws(ws_id, starter_rows=starter)
                st.session_state["df_lookup_seeded"] = True
            except Exception:
                pass
        df_lookup = _cached_read_lookup(ws_id, st.session_state["cache_bump"])
    except Exception as e:
        st.warning(f"Could not load lookup for '{ws_id}': {e}")
        df_lookup = pd.DataFrame(columns=LOOKUP_HEADERS)
else:
    st.info("No workspace selected — per-workspace lookup disabled.")
    df_lookup = pd.DataFrame(columns=LOOKUP_HEADERS)

# Optional: edit per-workspace lookup in sidebar
sidebar_header("⏱️", "Frequency lookup (workspace)")
with st.sidebar.expander("View/Edit Lookup", expanded=False):
    st.caption("Maps Level 1 values to Frequency and a monthly factor (workspace-specific).")
    df_lookup_edit = st.data_editor(
        df_lookup,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Level1": st.column_config.TextColumn("Level1"),
            "Frequency": st.column_config.SelectboxColumn("Frequency", options=FREQUENCY_CHOICES, required=False),
            "Factor_per_month": st.column_config.NumberColumn("Factor per month", step=0.0001, format="%.4f"),
        },
        key="lookup_editor",
        disabled=not bool(ws_id),
    )
    if ws_id and st.button("Save Lookup"):
        try:
            df_le = df_lookup_edit.copy()
            df_le["Level1"] = df_le["Level1"].astype(str).str.strip()
            df_le["Frequency"] = df_le["Frequency"].astype(str).str.strip()
            df_le["Factor_per_month"] = pd.to_numeric(df_le["Factor_per_month"], errors="coerce").fillna(1.0)
            write_lookup_for_ws(ws_id, df_le)
            st.success("Lookup saved.")
            st.session_state["cache_bump"] += 1
            _rerun()
        except Exception as e:
            st.error(f"Saving lookup failed: {e}")

# =======================================================================================
# MAIN TABS
# =======================================================================================
tab_setup, tab_overview, tab_settings = st.tabs(["🧭 Initialize", "📊 Summary", "⚙️ Settings"])

with tab_setup:
# Snapshots used in Initialize
    budget_df, tx_df, lookup_df = get_snapshots(df_raw)

    st.session_state.setdefault("__editor_rev", 0)
    # -----------------------------------
    # Budget table (editable) + preview
    # -----------------------------------
    section_header("📦", "Budget table", f"Source: {source_name}")

    editor_input_df = st.session_state.get("ws_budget_df", df_raw).copy()
    edited_df = st.data_editor(
        editor_input_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"setup_editor_{st.session_state['__editor_rev']}"
    )

    # Keep shared snapshot in sync with edits
    st.session_state["ws_budget_df"] = edited_df.copy()

    # -----------------------------------
    # Save budget to Workspace
    # -----------------------------------
    save_col1, _ = st.columns([1, 3])
    with save_col1:
        can_save = bool(ws_id)
        save_btn = st.button("Save Budget to Workspace", type="primary", disabled=not can_save)

    if not can_save:
        st.caption("Enter a Workspace ID in the sidebar to enable saving.")

    if save_btn and can_save:
        expected_version = st.session_state.get("ws_version", 0)
        df_to_save = edited_df.copy()
        try:
            new_version = write_workspace(ws_id, df_to_save, expected_version=expected_version)
            st.session_state["ws_version"] = new_version
            st.success(f"Saved workspace '{ws_id}' (version {new_version}).")
            st.session_state["cache_bump"] += 1
            _rerun()
        except VersionConflict as vc:
            st.warning(str(vc))
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Reload from Sheets"):
                    _rerun()
            with c2:
                if st.button("Overwrite anyway"):
                    try:
                        new_version = write_workspace(ws_id, df_to_save, expected_version=None)
                        st.session_state["ws_version"] = new_version
                        st.success(f"Overwrote workspace '{ws_id}' (version {new_version}).")
                        st.session_state["cache_bump"] += 1
                        _rerun()
                    except Exception as e:
                        st.error(f"Could not overwrite: {e}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # -----------------------------------
    # Guided Budget Builder (step-by-step rows) — REACTIVE (no st.form)
    # -----------------------------------
    st.divider()
    st.subheader("🧭 Guided Budget Builder")

    # Fresh snapshots (no mutation yet)
    _budget_df_snap, _tx_df_snap, _lookup_df_snap = get_snapshots(df_raw)

    # ---- Controls (start / factors / clear) ----
    # Initialize all guide-related keys exactly once
    st.session_state.setdefault("guide_active", False)
    st.session_state.setdefault("guide_rows", [])
    st.session_state.setdefault("show_factor_manager", False)

    def _on_toggle_change():
        # When turning the guide OFF, clear staged rows (optional)
        if not st.session_state.get("guide_active", False):
            st.session_state["guide_rows"].clear()

    if st.session_state.pop("__guide_reset_toggle", False):
        st.session_state["guide_active"] = False

    cols_head = st.columns([1, 1, 1, 2])
    with cols_head[0]:
        st.toggle(
            "Start Guide",
            key="guide_active",
            help="Create rows step-by-step.",
            on_change=_on_toggle_change
        )

    with cols_head[1]:
        if st.button("Manage Factors ⚙️", use_container_width=True):
            st.session_state["show_factor_manager"] = True

    with cols_head[2]:
        # Use .get(...) so we don't KeyError before init
        has_staged = bool(st.session_state.get("guide_rows", []))
        if st.button("Clear staged rows 🧹", use_container_width=True, disabled=not has_staged):
            st.session_state["guide_rows"].clear()
            st.success("Cleared staged rows.")

    # ===== When the guide is OFF, do not touch any frames =====
    if not st.session_state["guide_active"]:
        st.caption("💡 Turn **Start Guide** on to build rows step-by-step. Sankey charts stay visible while the guide is off.")
    else:
        # ===== Guide is ON: safe, local prep then persist only what’s needed =====
        budget_df = _budget_df_snap.copy()
        lookup_df = _lookup_df_snap.copy()

        # Ensure Subcategory, Category Universe, Account columns (guide-only)
        existing_subcat_col = _detect_subcategory_column(budget_df)
        subcategory_col = existing_subcat_col or "Subcategory"
        budget_df = _ensure_subcategory_column(budget_df, subcategory_col)

        cu_col = (
            st.session_state.get("cat_universe_col")
            or st.session_state.get("cu_col_saved")
            or "Category"
        )
        if cu_col not in budget_df.columns:
            budget_df[cu_col] = ""

        budget_account_col = _account_col_from_budget(budget_df) or "Account"
        if budget_account_col not in budget_df.columns:
            budget_df[budget_account_col] = ""

        # Persist guide-prepared frame for downstream widgets that rely on it
        st.session_state["ws_budget_df"] = budget_df

        # L1 factors cache
        if "guide_l1_factors" not in st.session_state:
            st.session_state["guide_l1_factors"] = _load_l1_factors_from_lookup(lookup_df)
        st.session_state.setdefault("guide_rows", [])

        # ===== Factor manager =====
        if st.session_state.get("show_factor_manager"):
            with st.expander("Manage Level 1 Factors", expanded=True):
                l1_factors = st.session_state["guide_l1_factors"]
                if not l1_factors:
                    st.info("No stored factors yet. Add some by creating rows or set them manually here.")
                editable = [{"Level1": k, "Factor_per_month": v} for k, v in sorted(l1_factors.items())]
                df_edit = pd.DataFrame(editable or [], columns=["Level1", "Factor_per_month"])
                df_edit = st.data_editor(
                    df_edit,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Level1": st.column_config.TextColumn("Income by / Source"),
                        "Factor_per_month": st.column_config.NumberColumn("Factor per month", step=0.05, format="%.2f"),
                    }
                )
                c1, c2 = st.columns([1, 2])
                with c1:
                    if st.button("Save Factors", type="primary", use_container_width=True):
                        new_map = {}
                        for _, r in df_edit.iterrows():
                            lvl = str(r["Level1"]).strip()
                            fac = float(pd.to_numeric(r["Factor_per_month"], errors="coerce") or 1.0)
                            if lvl:
                                new_map[lvl] = fac
                                lookup_df = _write_l1_factor(ws_id, lookup_df, lvl, fac)
                        st.session_state["guide_l1_factors"] = new_map
                        st.session_state["ws_lookup_df"] = lookup_df
                        st.success("Factors saved.")
                with c2:
                    if st.button("Close"):
                        st.session_state["show_factor_manager"] = False
                        _rerun()

        # ===== Helper: selectbox with instant "type new" input =====
        def pick_or_type(label: str, options: list[str], key: str, placeholder: str = "Type new…"):
            opts = ["— Type new —"] + options
            sel = st.selectbox(label, options=opts, key=f"{key}_sel")
            if sel == "— Type new —":
                typed = st.text_input(" ", key=f"{key}_typed", placeholder=placeholder, label_visibility="collapsed")
                return (typed.strip(), True) if typed else ("", True)
            return (sel, False)

        # ===== Guided inputs =====
        st.markdown("Configure one row at a time. You can insert staged rows into the table when ready.")

        existing_l1_vals = sorted(budget_df[l1].dropna().astype(str).unique().tolist()) if l1 in budget_df.columns else []
        existing_categories = sorted(budget_df[cu_col].dropna().astype(str).unique().tolist()) if cu_col in budget_df.columns else []
        existing_subs = sorted(budget_df[subcategory_col].dropna().astype(str).unique().tolist()) if subcategory_col in budget_df.columns else []
        existing_accounts = sorted(budget_df[budget_account_col].dropna().astype(str).unique().tolist()) if budget_account_col in budget_df.columns else []

        l1_col1, l1_col2 = st.columns([1, 1])
        with l1_col1:
            l1_val, l1_is_new = pick_or_type(
                "Income by / Source (Level 1)",
                list(dict.fromkeys(existing_l1_vals + DEFAULT_INCOME_SOURCES)),
                key="guide_l1",
                placeholder="e.g., Me / Partner / Side hustle"
            )
        with l1_col2:
            factor_known = bool(l1_val) and (l1_val in st.session_state["guide_l1_factors"])
            if l1_val and not factor_known:
                st.info(f"Set a **monthly factor** for '{l1_val}'. Asked once and reused.")
                presets = {"Monthly (1.0)": 1.0, "Bi-monthly (0.5)": 0.5, "Weekly (~4.3)": 4.3, "Custom": None}
                preset_choice = st.selectbox("Preset", list(presets.keys()), index=0, key="guide_factor_preset")
                custom_factor = None
                if presets[preset_choice] is None:
                    custom_factor = st.number_input("Custom factor per month", min_value=0.0, step=0.05, value=1.0, key="guide_factor_custom")
                proposed_factor = presets[preset_choice] if presets[preset_choice] is not None else custom_factor
                remember_factor = st.checkbox(f"Remember {proposed_factor or 1.0} for '{l1_val}'", value=True, key="guide_factor_remember")
            else:
                proposed_factor = st.session_state["guide_l1_factors"].get(l1_val, 1.0) if l1_val else 1.0
                remember_factor = False

        cat_col1, cat_col2 = st.columns([1, 1])
        with cat_col1:
            category_val, cat_is_new = pick_or_type(
                f"Category ({cu_col})",
                list(dict.fromkeys(flat_default_categories() + existing_categories)),
                key="guide_cat",
                placeholder="e.g., Groceries"
            )
        default_subs_for_cat = defaults_for(category_val) if category_val else []
        with cat_col2:
            sub_val, sub_is_new = pick_or_type(
                f"Subcategory ({subcategory_col})",
                list(dict.fromkeys(default_subs_for_cat + existing_subs)),
                key="guide_sub",
                placeholder="e.g., Pantry"
            )

        acc_col1, acc_col2 = st.columns([1, 1])
        with acc_col1:
            account_val, acc_is_new = pick_or_type(
                f"Account ({budget_account_col})",
                existing_accounts,
                key="guide_acct",
                placeholder="e.g., Checking"
            )
        with acc_col2:
            amount_val = st.number_input("Amount (positive number)", min_value=0.0, step=10.0, value=0.0, key="guide_amount")

        frc_col, note_col = st.columns([1, 1])
        with frc_col:
            freq_choice = st.selectbox("Frequency (helper)", ["Monthly", "Bi-monthly", "Weekly (~4.3)", "One-time", "Other"], index=0, key="guide_freq")
        with note_col:
            notes_val = st.text_input("Notes (optional)", placeholder="Add a short note", key="guide_notes")

        st.caption("Quick apply")
        qc1, qc2 = st.columns([1, 1])
        apply_account_all = qc1.checkbox("Apply Account to all of this Category", value=False, key="guide_apply_acct_all")
        apply_sub_all     = qc2.checkbox("Apply Subcategory to similar rows", value=False, key="guide_apply_sub_all")

        if st.button("Stage row ➕", type="primary", key="guide_stage_btn"):
            if not l1_val:
                st.error("Please provide an Income Source (Level 1).")
            elif not category_val:
                st.error("Please choose or type a Category.")
            elif amount_val <= 0:
                st.error("Amount must be greater than zero.")
            else:
                # Save factor once
                if l1_val and not factor_known and remember_factor and proposed_factor is not None:
                    lookup_df = _write_l1_factor(ws_id, lookup_df, l1_val, float(proposed_factor))
                    st.session_state["ws_lookup_df"] = lookup_df
                    st.session_state["guide_l1_factors"][l1_val] = float(proposed_factor)

                new_row = {col: "" for col in budget_df.columns}
                if l1 in new_row: new_row[l1] = l1_val
                new_row[cu_col] = category_val
                new_row[subcategory_col] = sub_val
                new_row[budget_account_col] = account_val
                if val_col in new_row:
                    new_row[val_col] = amount_val
                if "Frequency" in new_row:
                    new_row["Frequency"] = freq_choice
                if "Notes" in new_row:
                    note_text = notes_val
                    if freq_choice and freq_choice != "Monthly":
                        note_text = (note_text + f" [Freq: {freq_choice}]").strip()
                    new_row["Notes"] = note_text

                st.session_state["guide_rows"].append({
                    "row": new_row,
                    "apply_account_all": apply_account_all,
                    "apply_sub_all": apply_sub_all,
                    "l1": l1_val,
                    "category": category_val,
                    "subcategory": sub_val,
                    "account": account_val,
                })
                st.success("Row staged. You can stage more, or insert them into the table below.")

        # ---- Staged rows tray ----
        staged = st.session_state["guide_rows"]
        if staged:
            st.markdown("#### 🗂️ Staged rows")
            staged_df = pd.DataFrame([r["row"] for r in staged])
            if val_col not in staged_df.columns:
                staged_df[val_col] = 0.0
            st.dataframe(staged_df, use_container_width=True)

            i1, i2 = st.columns([1, 2])
            with i1:
                if st.button("Insert staged into Budget table ✅", type="primary", use_container_width=True, key="guide_insert_btn"):
                    target_df = st.session_state.get("ws_budget_df", df_raw.copy())
                    for col in staged_df.columns:
                        if col not in target_df.columns:
                            target_df[col] = ""
                    target_df = pd.concat([target_df, staged_df[target_df.columns]], ignore_index=True)
        
                    for item in staged:
                        if item["apply_account_all"] and item["category"]:
                            mask = target_df[cu_col].astype(str).eq(item["category"])
                            target_df.loc[mask, budget_account_col] = target_df.loc[mask, budget_account_col].replace("", item["account"])
                        if item["apply_sub_all"] and item["subcategory"]:
                            mask = target_df[cu_col].astype(str).eq(item["category"])
                            target_df.loc[mask, subcategory_col] = target_df.loc[mask, subcategory_col].replace("", item["subcategory"])
        
                    # ✅ update our data snapshot
                    st.session_state["ws_budget_df"] = target_df
        
                    # ✅ bump the editor revision so the data editor remounts with new data
                    st.session_state["__editor_rev"] += 1
        
                    # clear staged and refresh UI
                    st.session_state["guide_rows"] = []
                    st.session_state["__guide_reset_toggle"] = True
                    st.success("Inserted staged rows into the budget table.")
                    _rerun()

            with i2:
                st.caption("Tip: You can still edit cells in the main table and save to your workspace as usual.")

    # Small hint while guide is on
    if st.session_state["guide_active"]:
        st.caption("💡 While the guide is on, Sankey charts are hidden so you can focus on building rows.")
    if not st.session_state.get("guide_active", False):
        # =========================
        # Filters (Overview-only state keys)
        # =========================
        section_header("🪢 🔎", "Budget flow | Filters")

        # ---------- NEW: use snapshots captured earlier ----------
        budget_df = st.session_state.get("ws_budget_df", df_raw.copy())
        tx_df     = st.session_state.get("ws_tx_df", st.session_state.get("tx_staged", pd.DataFrame()))
        lookup_df = st.session_state.get("ws_lookup_df", pd.DataFrame(columns=LOOKUP_HEADERS))
        cu_col    = st.session_state.get("cu_col_saved") or st.session_state.get("cat_universe_col")
        # ---------------------------------------------------------

        OV_L1_KEY   = "ov_level1_selected"
        OV_L2_KEY   = "ov_level2_selected"
        OV_NORM_KEY = "ov_normalize"

        level1_values_all = sorted(budget_df[l1].dropna().astype(str).unique().tolist()) if l1 in budget_df.columns else []

        l1_signature = (l1, tuple(level1_values_all))
        prev_l1_signature = st.session_state.get("__ov_l1_signature")
        if (OV_L1_KEY not in st.session_state) or (prev_l1_signature != l1_signature):
            st.session_state["__ov_l1_signature"] = l1_signature
            st.session_state[OV_L1_KEY] = level1_values_all

        preset_l1 = st.session_state.get(OV_L1_KEY, level1_values_all)
        preset_l1 = [val for val in preset_l1 if val in level1_values_all]
        if not preset_l1 and level1_values_all:
            preset_l1 = level1_values_all
            st.session_state[OV_L1_KEY] = preset_l1

        col_f1, col_f3 = st.columns([3, 1])
        with col_f1:
            level1_selected = st.multiselect(
                "Filter Level 1 categories",
                options=level1_values_all,
                default=preset_l1,
                key=OV_L1_KEY,
                help="Filter the dataset by the mapped Level 1 column."
            )
            level1_selected = [val for val in st.session_state.get(OV_L1_KEY, []) if val in level1_values_all]
            if not level1_selected and level1_values_all:
                level1_selected = level1_values_all

        with col_f3:
            normalize_for_sankey = st.checkbox(
                "Normalize to monthly",
                value=True,
                key=OV_NORM_KEY,
                help="Convert values to monthly equivalents using the per-workspace lookup."
            )

        if l1 in budget_df.columns and l2 in budget_df.columns:
            df_for_l2 = budget_df[budget_df[l1].astype(str).isin(level1_selected)] if level1_selected else budget_df
            level2_values_all = sorted(df_for_l2[l2].dropna().astype(str).unique().tolist())
        else:
            level2_values_all = []

        l2_signature = (l1, l2, tuple(level2_values_all))
        prev_l2_signature = st.session_state.get("__ov_l2_signature")
        if (OV_L2_KEY not in st.session_state) or (prev_l2_signature != l2_signature):
            st.session_state["__ov_l2_signature"] = l2_signature
            st.session_state[OV_L2_KEY] = level2_values_all

        preset_l2 = st.session_state.get(OV_L2_KEY, level2_values_all)
        preset_l2 = [val for val in preset_l2 if val in level2_values_all]
        if not preset_l2 and level2_values_all:
            preset_l2 = level2_values_all
            st.session_state[OV_L2_KEY] = preset_l2

        level2_selected = st.multiselect(
            "Filter Level 2 categories",
            options=level2_values_all,
            default=preset_l2,
            key=OV_L2_KEY
        )
        level2_selected = [val for val in st.session_state.get(OV_L2_KEY, []) if val in level2_values_all]
        if not level2_selected and level2_values_all:
            level2_selected = level2_values_all

        # -----------------------------------
        # Budget Sankey (after table) — respects Overview filters & snapshots
        # -----------------------------------

        viz_df = budget_df.copy()
        if l1 in viz_df.columns and level1_selected:
            viz_df = viz_df[viz_df[l1].astype(str).isin(level1_selected)]
        if l2 in viz_df.columns and level2_selected:
            viz_df = viz_df[viz_df[l2].astype(str).isin(level2_selected)]

        val_to_use = val_col
        use_lookup_snap = not lookup_df.empty and (l1 in viz_df.columns)

        if normalize_for_sankey and use_lookup_snap:
            df_norm = viz_df.merge(
                lookup_df[["Level1", "Factor_per_month"]],
                how="left",
                left_on=l1,
                right_on="Level1"
            )
            df_norm["Factor_per_month"] = pd.to_numeric(df_norm["Factor_per_month"], errors="coerce").fillna(1.0)
            df_norm["_AmountMonthly"] = pd.to_numeric(df_norm[val_col], errors="coerce").fillna(0) * df_norm["Factor_per_month"]
            val_to_use = "_AmountMonthly"
        else:
            df_norm = viz_df.copy()

        try:
            sankey_data = prepare_sankey_data(df_norm, l1=l1, l2=l2, l3=l3, value_col=val_to_use, l4=l4)

            if not sankey_data["sources"]:
                st.info("No flows to display for the current filters.")
            elif chart_type == "Sankey (Plotly)":
                fig = make_sankey_figure(sankey_data, title=f"{l1} \u2192 {l2} \u2192 {l3}" + (f" \u2192 {l4}" if l4 else ""))
                fig.update_layout(template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if chart_type.startswith("Sankey (ECharts") and "ech_curveness" not in locals():
                    ech_curveness, ech_edge_font = 0.5, 11
                options = make_echarts_sankey_options(
                    sankey_data,
                    title=f"{l1} \u2192 {l2} \u2192 {l3}" + (f" \u2192 {l4}" if l4 else ""),
                    value_suffix=" $",
                    curveness=ech_curveness,
                    edge_font_size=int(ech_edge_font),
                    py_value_format=",.0f",
                )
                st_echarts(options=options, height="600px",theme=ECHR_THEME)
        except Exception as e:
            st.error(f"Could not render chart: {e}")
    else:
        st.info("Guided Builder is active — budget Sankey is hidden until you finish or turn it off.")
    if not st.session_state.get("guide_active", False):    
        # -----------------------------------
        # Transactions (map & classify) — uploader is in sidebar
        # -----------------------------------
        st.divider()
        section_header("📥", "Transactions — classify & stage")

        tx_col1, tx_col2 = st.columns([2, 1])

        with tx_col1:
            st.caption("Upload a transactions file in the **sidebar**, then map columns here.")
            tx_file = st.session_state.get("tx_file_latest")  # from sidebar

            # Transaction mappings UI
            tx_map = {}
            if tx_file is not None:
                try:
                    df_preview = read_table_any(tx_file)
                    tx_cols = list(df_preview.columns)
                except Exception:
                    tx_cols = []

                def _pick(lbl, candidates):
                    if not tx_cols:
                        return None
                    for c in candidates:
                        if c in tx_cols:
                            return c
                    return tx_cols[0]

                st.write("**Map transaction columns**")
                tx_map["Date"] = st.selectbox("Date", tx_cols, index=tx_cols.index(_pick("Date", ["Date","Posting Date","Transaction Date"])) if tx_cols else 0)
                tx_map["Description"] = st.selectbox("Description", tx_cols, index=tx_cols.index(_pick("Description", ["Description","Details","Narrative"])) if tx_cols else 0)
                tx_map["Merchant"] = st.selectbox("Merchant (optional)", ["(none)"] + tx_cols, index=0)
                tx_map["Merchant"] = None if tx_map["Merchant"] == "(none)" else tx_map["Merchant"]
                tx_map["Account"] = st.selectbox("Account", tx_cols, index=tx_cols.index(_pick("Account", ["Account","Card","Account Name"])) if tx_cols else 0)
                tx_map["Amount"] = st.selectbox("Amount", tx_cols, index=tx_cols.index(_pick("Amount", ["Amount","Value","Debit","Credit"])) if tx_cols else 0)

                # Period selector (majority month)
                if tx_cols:
                    try:
                        dt_preview = pd.to_datetime(df_preview[tx_map["Date"]], errors="coerce")
                        ym_preview = dt_preview.dt.to_period("M").astype(str)
                        counts = ym_preview.value_counts()
                        period_suggest = counts.idxmax() if not counts.empty else pd.Timestamp.today().to_period("M").strftime("%Y-%m")
                        period_options = sorted(ym_preview.dropna().unique().tolist())
                        if period_suggest not in period_options:
                            period_options = sorted(set(period_options + [period_suggest]))
                        # --- Code to add the next period starts here ---
                        if period_options:
                            # Get the last period in the sorted list
                            last_period_str = period_options[-1]
                            # Convert the string to a pandas Period object
                            last_period = pd.Period(last_period_str, freq='M')
                            # Advance the period by one month
                            next_period = last_period + 1
                            # Convert the next period back to the 'YYYY-MM' string format
                            next_period_str = next_period.strftime("%Y-%m")
                            # Add the next period to the options list
                            if next_period_str not in period_options:
                                period_options.append(next_period_str)
                        # --- Code to add the next period ends here ---
                    except Exception:
                        period_suggest = pd.Timestamp.today().to_period("M").strftime("%Y-%m")
                        period_options = [period_suggest]
                else:
                    period_suggest = pd.Timestamp.today().to_period("M").strftime("%Y-%m")
                    period_options = [period_suggest]

                selected_period = st.selectbox(
                    "Period for this upload (YYYY-MM)",
                    options=period_options or [period_suggest],
                    index=(period_options.index(period_suggest) if period_suggest in period_options else 0),
                    key="tx_period_select",
                    help="We suggest the calendar month (YYYY-MM) that appears most often in the file’s dates. You can override it."
                )

                if st.button("Parse & stage transactions", type="primary"):
                    try:
                        df_tx = read_transactions_file(tx_file, tx_map)

                        if "Notes" not in df_tx.columns:
                            df_tx["Notes"] = ""

                        sel_period = st.session_state.get("tx_period_select")
                        if not sel_period:
                            dt_tx = pd.to_datetime(df_tx["Date"], errors="coerce")
                            ym_tx = dt_tx.dt.to_period("M").astype(str)
                            counts_tx = ym_tx.value_counts()
                            sel_period = counts_tx.idxmax() if not counts_tx.empty else pd.Timestamp.today().to_period("M").strftime("%Y-%m")
                        df_tx["Period"] = str(sel_period)

                        df_tx = _ensure_tx_id(df_tx)

                        cu_col = st.session_state.get("cat_universe_col") or (read_category_universe(ws_id) if ws_id else None)
                        if cu_col in edited_df.columns:
                            category_universe = sorted(
                                edited_df[cu_col].dropna().astype(str).str.strip().unique().tolist()
                            )
                        else:
                            category_universe = []

                        if "AssignedCategory" not in df_tx.columns:
                            df_tx["AssignedCategory"] = ""

                        if ws_id:
                            try:
                                df_mem = read_merchants(ws_id)  # MerchantKey, AssignedCategory
                            except Exception:
                                df_mem = pd.DataFrame(columns=["MerchantKey", "AssignedCategory"])
                        else:
                            df_mem = pd.DataFrame(columns=["MerchantKey", "AssignedCategory"])

                        if not df_mem.empty:
                            mem_map = (
                                df_mem.dropna()
                                .assign(
                                    MerchantKey=lambda d: d["MerchantKey"].astype(str)
                                        .str.lower()
                                        .str.replace("/", " ", regex=False)
                                        .str.replace(r"\s+", " ", regex=True)
                                        .str.strip(),
                                    AssignedCategory=lambda d: d["AssignedCategory"].astype(str).str.strip(),
                                )
                                .drop_duplicates(subset=["MerchantKey"], keep="last")
                                .set_index("MerchantKey")["AssignedCategory"]
                                .to_dict()
                            )
                        else:
                            mem_map = {}

                        mk = df_tx["Merchant"].astype(str).fillna("") \
                                .str.lower() \
                                .str.replace("/", " ", regex=False) \
                                .str.replace(r"\s+", " ", regex=True) \
                                .str.strip()

                        proposed_mem = mk.map(mem_map).fillna("")
                        mask_empty = df_tx["AssignedCategory"].astype(str).fillna("").str.strip().isin(["","Uncategorized"])
                        df_tx.loc[mask_empty & proposed_mem.ne(""), "AssignedCategory"] = proposed_mem[mask_empty & proposed_mem.ne("")]

                        if category_universe:
                            joined_text = (
                                df_tx["Description"].fillna("").astype(str) + " " +
                                df_tx["Merchant"].fillna("").astype(str)
                            ).str.lower().str.replace("/", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

                            still_empty = df_tx["AssignedCategory"].astype(str).fillna("").str.strip().eq("")
                            idxs = still_empty[still_empty].index.tolist()
                            if idxs:
                                for idx in idxs:
                                    label, score = _suggest_category_universe_with_score(joined_text.loc[idx], category_universe, cutoff=60)
                                    if label:
                                        df_tx.at[idx, "AssignedCategory"] = label

                        st.session_state["tx_staged"] = df_tx
                        st.session_state["ws_tx_df"] = df_tx.copy()  # keep Overview current
                        st.session_state["tx_universe"] = category_universe
                        st.success(f"Parsed {len(df_tx):,} transactions. Scroll down to edit & save.")

                    except Exception as e:
                        st.error(f"Parse failed: {e}")
                        print(f"[PARSE-ERR] {e}")

        with tx_col2:
            st.caption("Load/save from your workspace")
            if ws_id:
                if st.button("Load transactions from workspace"):
                    try:
                        txws = read_transactions(ws_id)
                        st.session_state["tx_staged"] = txws
                        st.session_state["ws_tx_df"] = txws.copy()  # keep Overview current
                        cu_col = st.session_state.get("cat_universe_col") or (read_category_universe(ws_id) if ws_id else None)
                        if cu_col in edited_df.columns:
                            st.session_state["tx_universe"] = sorted(edited_df[cu_col].dropna().astype(str).str.strip().unique().tolist())
                        else:
                            st.session_state["tx_universe"] = []
                        st.success("Transactions loaded.")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
            else:
                st.info("Enter a Workspace ID to enable load/save.")

        # -----------------------------------
        # Transactions table (sortable) + filter + save/apply
        # -----------------------------------
        if "tx_staged" in st.session_state:
            st.divider()
            section_header("📋", "Transactions table")

            base_df = st.session_state["tx_staged"].copy()
            universe = st.session_state.get("tx_universe", [])

            for col in ["AssignedCategory","Period","Notes"]:
                if col not in base_df.columns:
                    base_df[col] = ""

            with st.expander("Filter rows (Transactions)", expanded=False):
                q_tx = st.text_input("Search text (all columns)", key="q_tx").strip().lower()
                col_filter_tx = st.selectbox("Filter by column (optional)", options=["(none)"] + list(base_df.columns), index=0, key="col_filter_tx")
                val_filter_tx = ""
                if col_filter_tx and col_filter_tx != "(none)":
                    val_filter_tx = st.text_input(f"Show rows where **{col_filter_tx}** contains:", key="val_filter_tx").strip().lower()

            col_cfg = {
                "AssignedCategory": st.column_config.SelectboxColumn(
                    "AssignedCategory",
                    options=universe or ["Uncategorized"],
                    required=False
                ),
                "Period": st.column_config.TextColumn("Period (YYYY-MM)"),
                "Notes": st.column_config.TextColumn("Notes"),
            }

            with st.form("tx_edit_form", clear_on_submit=False):
                st.caption("Tip: edit as many cells as you want; changes apply on submit.")
                df_tx_form = st.data_editor(
                    base_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="tx_editor",
                    column_config=col_cfg
                )
                fcol1, fcol2 = st.columns([1,1])
                save_clicked = fcol1.form_submit_button("Save transactions to workspace", type="primary", use_container_width=True)
                apply_clicked = fcol2.form_submit_button("Apply edits (keep local)", use_container_width=True)

            df_tx_preview = df_tx_form.copy()
            if q_tx:
                df_tx_preview = df_tx_preview[df_tx_preview.apply(lambda r: q_tx in str(r.values).lower(), axis=1)]
            if col_filter_tx and col_filter_tx != "(none)" and val_filter_tx:
                df_tx_preview = df_tx_preview[df_tx_preview[col_filter_tx].astype(str).str.lower().str.contains(val_filter_tx, na=False)]

    #        st.caption("Preview (filtered)")
    #        st.dataframe(df_tx_preview, use_container_width=True)

            if apply_clicked:
                st.session_state["tx_staged"] = df_tx_form
                st.session_state["ws_tx_df"] = df_tx_form.copy()  # keep Overview current
                st.success("Edits applied locally.")
                st.rerun()

            if save_clicked and ws_id:
                try:
                    df_to_save = df_tx_form.copy()

                    uni = st.session_state.get("tx_universe", [])
                    if uni:
                        bad = ~df_to_save["AssignedCategory"].isin(["", *uni])
                        n_bad = int(bad.sum())
                        if n_bad:
                            st.warning(f"{n_bad} rows have categories not in the Category Universe; they were set to 'Uncategorized'.")
                            df_to_save.loc[bad, "AssignedCategory"] = "Uncategorized"

                    df_to_save = _ensure_tx_id(df_to_save)

                    try:
                        existing = read_transactions(ws_id)
                    except Exception:
                        existing = pd.DataFrame()

                    if existing is None or existing.empty:
                        combined = df_to_save.copy()
                        prev_rows = 0
                    else:
                        existing = _ensure_tx_id(existing.copy())
                        prev_rows = len(existing)
                        combined = pd.concat([existing, df_to_save], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["TX_ID"], keep="last")

                    write_transactions(ws_id, combined)

                    added_rows = max(0, len(combined) - prev_rows)
                    st.info(f"Appended {added_rows} new transactions. Total now: {len(combined):,}.")

                    try:
                        if {"Merchant", "AssignedCategory"}.issubset(df_to_save.columns):
                            df_mem = df_to_save[["Merchant", "AssignedCategory"]].copy()
                            df_mem["AssignedCategory"] = df_mem["AssignedCategory"].astype(str).str.strip()
                            df_mem = df_mem[(df_mem["AssignedCategory"] != "") & (df_mem["AssignedCategory"].str.lower() != "uncategorized")]
                            df_mem["MerchantKey"] = (
                                df_mem["Merchant"].astype(str)
                                .str.lower()
                                .str.replace("/", " ", regex=False)
                                .str.replace(r"\s+", " ", regex=True)
                                .str.strip()
                            )
                            df_mem = df_mem[df_mem["MerchantKey"] != ""]
                            df_mem = df_mem[["MerchantKey", "AssignedCategory"]].drop_duplicates(subset=["MerchantKey"], keep="last")

                            base_mem = read_merchants(ws_id)
                            if base_mem is None or base_mem.empty:
                                merged = df_mem
                            else:
                                base_mem = base_mem[["MerchantKey", "AssignedCategory"]].dropna(how="any")
                                base_mem["MerchantKey"] = (
                                    base_mem["MerchantKey"].astype(str)
                                    .str.lower()
                                    .str.replace("/", " ", regex=False)
                                    .str.replace(r"\s+", " ", regex=True)
                                    .str.strip()
                                )
                                base_mem["AssignedCategory"] = base_mem["AssignedCategory"].astype(str).str.strip()
                                base_mem = base_mem[(base_mem["MerchantKey"] != "") & (base_mem["AssignedCategory"] != "")]
                                base_mem = base_mem.drop_duplicates(subset=["MerchantKey"], keep="last")
                                merged = pd.concat([base_mem, df_mem], ignore_index=True)

                            merged = merged.dropna().drop_duplicates(subset=["MerchantKey"], keep="last")
                            write_merchants(ws_id, merged)
                    except Exception as e:
                        print(f"[MEM-WRITE-ERR] {e}")

                    st.session_state["tx_staged"] = df_tx_form
                    st.session_state["ws_tx_df"] = df_tx_form.copy()  # keep Overview current
                    st.success("Transactions saved to workspace.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

            c2, c3 = st.columns([1,1])
            with c2:
                if st.button("Clear staged transactions"):
                    st.session_state.pop("tx_staged", None)
                    st.session_state.pop("tx_universe", None)
                    st.session_state["ws_tx_df"] = pd.DataFrame()  # keep Overview current
                    st.info("Cleared staged transactions.")
                    st.rerun()

            with c3:
                overwrite_filled = st.checkbox(
                    "Overwrite filled",
                    value=False,
                    help="If checked, memory + fuzzy will recompute for all rows. If off, only empty AssignedCategory cells are filled."
                )
                try:
                    df_tx_prev = st.session_state.get("tx_staged", pd.DataFrame()).copy()
                    universe = st.session_state.get("tx_universe", [])
                    if df_tx_prev.empty or not universe:
                        st.info("No staged transactions or Category Universe to compute against.")
                    else:
                        if "AssignedCategory" not in df_tx_prev.columns:
                            df_tx_prev["AssignedCategory"] = ""
                        cur_assigned = df_tx_prev["AssignedCategory"].astype(str).fillna("").str.strip()
                        mask_target = pd.Series(True, index=df_tx_prev.index) if overwrite_filled else cur_assigned.isin(["","Uncategorized"])
                        total_target = int(mask_target.sum())

                        if ws_id:
                            try:
                                df_mem_prev = read_merchants(ws_id)
                            except Exception:
                                df_mem_prev = pd.DataFrame(columns=["MerchantKey","AssignedCategory"])
                        else:
                            df_mem_prev = pd.DataFrame(columns=["MerchantKey","AssignedCategory"])

                        mem_map = {}
                        if not df_mem_prev.empty:
                            mem_map = (
                                df_mem_prev.dropna()
                                .assign(
                                    MerchantKey=lambda d: d["MerchantKey"].astype(str)
                                        .str.lower()
                                        .str.replace("/", " ", regex=False)
                                        .str.replace(r"\s+", " ", regex=True)
                                        .str.strip(),
                                    AssignedCategory=lambda d: d["AssignedCategory"].astype(str).str.strip()
                                )
                                .drop_duplicates(subset=["MerchantKey"], keep="last")
                                .set_index("MerchantKey")["AssignedCategory"]
                                .to_dict()
                            )

                        local_pairs = df_tx_prev[["Merchant", "AssignedCategory"]].copy()
                        local_pairs["AssignedCategory"] = local_pairs["AssignedCategory"].astype(str).str.strip()
                        local_pairs = local_pairs[(local_pairs["AssignedCategory"] != "") & (local_pairs["AssignedCategory"].str.lower() != "uncategorized")]
                        local_pairs["MerchantKey"] = (
                            local_pairs["Merchant"].astype(str)
                            .str.lower()
                            .str.replace("/", " ", regex=False)
                            .str.replace(r"\s+", " ", regex=True)
                            .str.strip()
                        )
                        local_pairs = local_pairs[local_pairs["MerchantKey"] != ""]
                        local_map = (
                            local_pairs.drop_duplicates(subset=["MerchantKey"], keep="last")
                            .set_index("MerchantKey")["AssignedCategory"].to_dict()
                        )
                        if local_map:
                            mem_map.update(local_map)

                        mk = df_tx_prev["Merchant"].astype(str) \
                                .str.lower() \
                                .str.replace("/", " ", regex=False) \
                                .str.replace(r"\s+", " ", regex=True) \
                                .str.strip()
                        proposed_mem = mk.map(mem_map).fillna("")
                        mem_fill_mask = mask_target & proposed_mem.ne("")
                        memory_hits = int(mem_fill_mask.sum())

                        after_mem_assigned = cur_assigned.where(~mem_fill_mask, proposed_mem)

                        fuzzy_hits = 0
                        unmatched_mask = mask_target & after_mem_assigned.eq("")
                        if int(unmatched_mask.sum()) > 0:
                            joined_text_prev = (
                                df_tx_prev["Description"].fillna("").astype(str) + " " +
                                df_tx_prev["Merchant"].fillna("").astype(str)
                            ).str.lower().str.replace("/", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

                            idxs = unmatched_mask[unmatched_mask].index.tolist()
                            fuzzy_labels = {}
                            for idx in idxs:
                                label, score = _suggest_category_universe_with_score(
                                    joined_text_prev.loc[idx], universe, cutoff=60
                                )
                                if label:
                                    fuzzy_labels[idx] = label
                            fuzzy_hits = len(fuzzy_labels)
                        else:
                            fuzzy_labels = {}

                        remaining = max(0, total_target - memory_hits - fuzzy_hits)
                        st.info(
                            f"Recompute (dry-run): target rows = {total_target} • "
                            f"memory matches = {memory_hits} • fuzzy matches = {fuzzy_hits} • remaining = {remaining}"
                        )

                        if st.button("Apply suggestions"):
                            df_tx_apply = df_tx_prev.copy()
                            assigned_now = df_tx_apply["AssignedCategory"].astype(str).fillna("").str.strip()
                            mask_tgt_apply = pd.Series(True, index=df_tx_apply.index) if overwrite_filled else assigned_now.isin(["", "Uncategorized"])

                            mk_apply = df_tx_apply["Merchant"].astype(str) \
                                        .str.lower() \
                                        .str.replace("/", " ", regex=False) \
                                        .str.replace(r"\s+", " ", regex=True) \
                                        .str.strip()
                            proposed_mem_apply = mk_apply.map(mem_map).fillna("")
                            mem_fill_apply = mask_tgt_apply & proposed_mem_apply.ne("")
                            mem_fill_apply_count = int(mem_fill_apply.sum())
                            df_tx_apply.loc[mem_fill_apply, "AssignedCategory"] = proposed_mem_apply[mem_fill_apply]

                            still_empty_apply = mask_tgt_apply & df_tx_apply["AssignedCategory"].astype(str).fillna("").str.strip().eq("")
                            applied_fuzzy = 0
                            if int(still_empty_apply.sum()) > 0:
                                joined_text_apply = (
                                    df_tx_apply["Description"].fillna("").astype(str) + " " +
                                    df_tx_apply["Merchant"].fillna("").astype(str)
                                ).str.lower().str.replace("/", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

                                for idx in still_empty_apply[still_empty_apply].index.tolist():
                                    label, score = _suggest_category_universe_with_score(
                                        joined_text_apply.loc[idx], universe, cutoff=60
                                    )
                                    if label:
                                        df_tx_apply.at[idx, "AssignedCategory"] = label
                                        applied_fuzzy += 1

                            st.session_state["tx_staged"] = df_tx_apply
                            st.session_state["ws_tx_df"] = df_tx_apply.copy()  # keep Overview current
                            st.success(f"Suggestions recomputed (memory {mem_fill_apply_count}, fuzzy {applied_fuzzy}).")
                            st.rerun()

                except Exception as e:
                    st.error(f"Preview/apply failed: {e}")

    # =========================
    # Actuals (Transactions) Sankey — Expense -> AssignedCategory -> Merchant
    # =========================
if not st.session_state.get("guide_active", False):
    if tx_df is not None and not pd.DataFrame(tx_df).empty:
        section_header("💳", "Actuals flow (Sankey)")
        try:
            df_tx_show = tx_df.copy()
            df_tx_show["Amount"] = pd.to_numeric(df_tx_show.get("Amount", 0), errors="coerce").fillna(0)

            if "AssignedCategory" not in df_tx_show.columns:
                df_tx_show["AssignedCategory"] = "Uncategorized"
            else:
                df_tx_show["AssignedCategory"] = df_tx_show["AssignedCategory"].astype(str).str.strip().replace({"": "Uncategorized"})

            # Find a Merchant-like column (do NOT use 'Period' as a candidate)
            merchant_col = None
            for cand in ["Period","Merchant", "Payee", "Vendor", "Description"]:
                if cand in df_tx_show.columns:
                    merchant_col = cand
                    break

            if merchant_col is None:
                st.warning("Could not find a Period/Merchant/Payee/Vendor/Description column — cannot render Actuals Sankey.")
            else:
                df_tx_show["L1_actual"] = "Expense"

                sankey_tx = prepare_sankey_data(
                    df_tx_show.rename(columns={
                        "L1_actual": "L1_actual",
                        merchant_col: "Period",
                        "AssignedCategory": "AssignedCategory",
                        "Amount": "Amount",
                    }),
                    l1="L1_actual",
                    l2="Period",
                    l3="AssignedCategory",
                    value_col="Amount",
                )

                title_tx = "Expense \u2192 AssignedCategory \u2192 Period"
                if chart_type == "Sankey (Plotly)":
                    fig_tx = make_sankey_figure(sankey_tx, title=title_tx)
                    fig_tx.update_layout(template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_tx, use_container_width=True)
                else:
                    if chart_type.startswith("Sankey (ECharts") and "ech_curveness" not in locals():
                        ech_curveness, ech_edge_font = 0.5, 11
                    opts_tx = make_echarts_sankey_options(
                        sankey_tx,
                        title=title_tx,
                        value_suffix=" $",
                        curveness=ech_curveness,
                        edge_font_size=int(ech_edge_font),
                        py_value_format=",.0f",
                    )
                    st_echarts(opts_tx, height="600px",theme=ECHR_THEME)

        except Exception as e:
            st.error(f"Could not render Actuals Sankey: {e}")

with tab_overview:
# Snapshots used in Summary
    budget_df, tx_df, lookup_df = get_snapshots(df_raw)

    # -----------------------------------
    # Period Summary + Budget vs Actuals + Payments
    # -----------------------------------
    st.divider()
    section_header("🗓️", "Period summary (Actuals)")

    df_tx_summary_src = tx_df.copy() if tx_df is not None else pd.DataFrame()
    if df_tx_summary_src is None or df_tx_summary_src.empty:
        st.info("No transactions found yet. Upload and save (or stage) some transactions to see summaries.")
    else:
        df_tx_sum = df_tx_summary_src.copy()
        for col in ["AssignedCategory", "Amount"]:
            if col not in df_tx_sum.columns:
                df_tx_sum[col] = "" if col == "AssignedCategory" else 0

        if "Period" not in df_tx_sum.columns or df_tx_sum["Period"].isna().all():
            try:
                dt = pd.to_datetime(df_tx_sum["Date"], errors="coerce")
                df_tx_sum["Period"] = dt.dt.to_period("M").astype(str)
            except Exception:
                df_tx_sum["Period"] = ""

        df_tx_sum["AssignedCategory"] = df_tx_sum["AssignedCategory"].astype(str).str.strip().replace({"": "Uncategorized"})
        df_tx_sum["Period"] = df_tx_sum["Period"].astype(str).str.strip()
        df_tx_sum["Amount"] = _to_numeric_safe(df_tx_sum["Amount"])

        periods_all = [p for p in sorted(df_tx_sum["Period"].dropna().unique().tolist()) if p != ""]
        if not periods_all:
            st.info("No valid Period values found in transactions.")
        else:
            col_ps1, col_ps2 = st.columns([2, 1])
            with col_ps1:
                period_selected = st.selectbox(
                    "Choose a Period (YYYY-MM)",
                    options=periods_all,
                    index=max(0, len(periods_all) - 1),
                    help="Select which month to summarize."
                )
            with col_ps2:
                show_zero_rows = st.checkbox("Show categories with zero total", value=False)

            df_period = df_tx_sum[df_tx_sum["Period"] == period_selected].copy()
                    
        # --- Summary cards (Total spend • # Transactions • % Categorized) ---
        # Uses the currently selected period (df_period)
        st.caption("Category totals for selected period")
        try:
            # Safe numeric
            df_period["Amount"] = pd.to_numeric(df_period.get("Amount", 0), errors="coerce").fillna(0)

            # Totals
            total_spend = float(df_period["Amount"].sum())
            tx_count = int(len(df_period))

            # % categorized (non-empty & not 'Uncategorized')
            cat_col = df_period.get("AssignedCategory")
            if cat_col is None:
                pct_categorized = 0.0
            else:
                cat_clean = cat_col.astype(str).str.strip().str.lower()
                is_categorized = (cat_clean != "") & (cat_clean != "uncategorized")
                pct_categorized = (is_categorized.mean() * 100.0) if tx_count > 0 else 0.0

            # Small helper to render a clean card
            def _card(col, icon, label, value_str):
                col.markdown(
                    f"""
                    <div style="
                        padding:14px 16px;
                        border-radius:16px;
                        background:#f8fafc;
                        border:1px solid #e2e8f0;
                    ">
                    <div style="font-size:14px;color:#64748b;display:flex;gap:.5rem;align-items:center;">
                        <span style="font-size:18px;">{icon}</span>
                        <span>{label}</span>
                    </div>
                    <div style="font-size:28px;font-weight:700;margin-top:6px;color:#0f172a;">
                        {value_str}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c1, c2, c3 = st.columns(3)
            _card(c1, "💸", "Total spend", f"${total_spend:,.0f}")
            _card(c2, "🧾", "Transactions", f"{tx_count:,}")
            _card(c3, "✅", "Categorized", f"{pct_categorized:,.0f}%")
        except Exception as _cards_err:
            st.caption(f"Summary cards unavailable for this period: {_cards_err}")

            summary = (
                df_period
                .groupby("AssignedCategory", dropna=False, as_index=False)["Amount"]
                .sum()
                .sort_values("Amount", ascending=False)
            )

            if show_zero_rows:
                universe = st.session_state.get("tx_universe", [])
                if universe:
                    zero_df = pd.DataFrame({"AssignedCategory": list(set(universe) - set(summary["AssignedCategory"]))})
                    zero_df["Amount"] = 0.0
                    summary = pd.concat([summary, zero_df], ignore_index=True).sort_values("Amount", ascending=False)

            total_amt = float(summary["Amount"].sum())
            st.markdown(f"**Period:** {period_selected} &nbsp;&nbsp;|&nbsp;&nbsp; **Total spend:** {total_amt:,.2f}")

            st.dataframe(
                summary.rename(columns={"AssignedCategory": "Category", "Amount": "Total"}),
                use_container_width=True
            )

            csv_sum = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download summary (CSV)",
                data=csv_sum,
                file_name=f"summary_{period_selected}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Budget vs Actuals bar chart
    section_header("📈", "Budget vs Actuals (by category)")
    cu_selected = st.session_state.get("cat_universe_col") or st.session_state.get("cu_col_saved")
    if not cu_selected or cu_selected not in budget_df.columns:
        st.warning("Category Universe column is not set or not found in the workbook. Save it in the sidebar first.")
    else:
        budget_cat_col = cu_selected
        col_ctl1, col_ctl2 = st.columns([1, 1])
        with col_ctl1:
            normalize_budget = st.checkbox(
                "Normalize budget to monthly (workspace lookup)",
                value=True,
                help="Uses Level1 → Factor_per_month from the per-workspace lookup table."
            )
        with col_ctl2:
            hide_zero = st.checkbox(
                "Hide zero-total categories",
                value=True,
                help="Hide rows where both Budget and Actuals are zero."
            )

        budget_base = budget_df.copy()
        budget_base["_val"] = pd.to_numeric(budget_base[val_col], errors="coerce").fillna(0)
        use_lookup_b = not lookup_df.empty and (l1 in budget_base.columns)

        if normalize_budget and use_lookup_b:
            bnorm = budget_base.merge(
                lookup_df[["Level1", "Factor_per_month"]],
                how="left",
                left_on=l1,
                right_on="Level1"
            )
            bnorm["Factor_per_month"] = pd.to_numeric(bnorm["Factor_per_month"], errors="coerce").fillna(1.0)
            bnorm["_val_monthly"] = bnorm["_val"] * bnorm["Factor_per_month"]
            value_col_for_budget = "_val_monthly"
            budget_agg_src = bnorm
        else:
            value_col_for_budget = "_val"
            budget_agg_src = budget_base

        if budget_cat_col in budget_agg_src.columns:
            bud = (
                budget_agg_src
                .groupby(budget_cat_col, as_index=False)[value_col_for_budget]
                .sum()
                .rename(columns={budget_cat_col: "Category", value_col_for_budget: "Budget"})
            )
        else:
            bud = pd.DataFrame(columns=["Category", "Budget"])

        if 'df_period' in locals() and not df_period.empty:
            act = (
                df_period.copy()
                .assign(Amount=pd.to_numeric(df_period["Amount"], errors="coerce").fillna(0))
                .groupby("AssignedCategory", as_index=False)["Amount"]
                .sum()
                .rename(columns={"AssignedCategory": "Category", "Amount": "Actuals"})
            )
        else:
            act = pd.DataFrame(columns=["Category", "Actuals"])

        combo = bud.merge(act, on="Category", how="outer").fillna({"Budget": 0, "Actuals": 0})
        if hide_zero and not combo.empty:
            combo = combo[(combo["Actuals"] != 0)]

        if not combo.empty:
            combo["_sort"] = combo[["Budget", "Actuals"]].max(axis=1)
            combo = combo.sort_values("_sort", ascending=False).drop(columns=["_sort"])

        st.dataframe(combo, use_container_width=True)

        try:
            import plotly.express as px
            melt = combo.melt(
                id_vars=["Category"], value_vars=["Budget", "Actuals"],
                var_name="Type", value_name="Value"
            )
            color_map = {'Budget':'rgba(136, 204, 238,0.7)','Actuals':'rgba(239, 85, 59, 0.7)'}
            fig_bar = px.bar(
                melt,
                x="Value",
                y="Category",
                color="Type",
                color_discrete_map=color_map,
                barmode="group",
                orientation="h",
                title=f"Budget vs Actuals — {locals().get('period_selected','(no period)')}",
            )
            fig_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="", height=500)
            fig_bar.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render Plotly bar chart: {e}")

    # ---------------------------------------------------------
    # Payments by Account (route Actuals to budgeted bank accounts)
    # ---------------------------------------------------------
    section_header("🏦", "Payments by account")

    def _find_account_column(df: pd.DataFrame) -> str | None:
        """
        Try to locate the budget account column in the workbook.
        Priority: exact case-insensitive match -> contains match.
        """
        if df is None or df.empty:
            return None

        priority = ["Bank Account", "Account", "Destination"]
        cols = list(df.columns)

        # 1) exact case-insensitive match
        for target in priority:
            for c in cols:
                if str(c).strip().lower() == target.lower():
                    return c

        # 2) substring case-insensitive match (fallback)
        for target in priority:
            for c in cols:
                if target.lower() in str(c).strip().lower():
                    return c

        return None

    cu_selected = st.session_state.get("cat_universe_col") or st.session_state.get("cu_col_saved")
    if not cu_selected or cu_selected not in budget_df.columns:
        st.warning("Category Universe column is not set or not found in the workbook, so we can’t map categories to accounts.")
    else:
        budget_acct_col = _find_account_column(budget_df)

        if not budget_acct_col:
            st.warning(
                "Could not find a bank account column in the workbook. "
                "Add a column named 'Bank Account', 'Account', or 'Destination' (any case)."
            )
        elif budget_acct_col == cu_selected:
            st.warning(
                "Payments mapping needs two different columns: one for Category Universe "
                f"(`{cu_selected}`) and one for Account (`{budget_acct_col}`). "
                "Right now they resolve to the same column."
            )
        else:
            st.markdown(
                f"<span style='display:inline-block;padding:.2rem .6rem;border-radius:999px;"
                f"background:#eef2ff;color:#3730a3;font-weight:600;font-size:0.85rem;'>"
                f"Using account column: {budget_acct_col}</span>",
                unsafe_allow_html=True
            )
            cpa1, cpa2 = st.columns([1, 1])
            with cpa1:
                normalize_for_routing = st.checkbox(
                    "Normalize budget to monthly for routing",
                    value=True,
                    help="Use the workspace lookup (Level1 → Factor_per_month) to choose the dominant account per category."
                )
            with cpa2:
                hide_zero_payments = st.checkbox("Hide zero-amount accounts", value=True)

            df_period_safe = locals().get("df_period", pd.DataFrame(columns=["AssignedCategory", "Amount"]))
            actuals_cat = (
                df_period_safe.copy()
                .assign(Amount=lambda d: pd.to_numeric(d.get("Amount", pd.Series(dtype=float)), errors="coerce").fillna(0))
                .groupby("AssignedCategory", as_index=False)["Amount"]
                .sum()
                .rename(columns={"AssignedCategory": "Category", "Amount": "Actuals"})
            )

            budget_map_base = budget_df.copy()
            budget_map_base["_val"] = pd.to_numeric(budget_map_base[val_col], errors="coerce").fillna(0)

            use_lookup = not lookup_df.empty and (l1 in budget_map_base.columns)
            if normalize_for_routing and use_lookup:
                bnorm = budget_map_base.merge(
                    lookup_df[["Level1", "Factor_per_month"]],
                    how="left",
                    left_on=l1,
                    right_on="Level1"
                )
                bnorm["Factor_per_month"] = pd.to_numeric(bnorm["Factor_per_month"], errors="coerce").fillna(1.0)
                bnorm["_val_monthly"] = bnorm["_val"] * bnorm["Factor_per_month"]
                route_val_col = "_val_monthly"
                map_src = bnorm
            else:
                route_val_col = "_val"
                map_src = budget_map_base

            if cu_selected not in map_src.columns or budget_acct_col not in map_src.columns:
                st.warning("Workbook doesn’t contain the expected Category Universe and Account columns to map payments.")
            else:
                pair = (
                    map_src
                    .groupby([cu_selected, budget_acct_col], as_index=False)[route_val_col]
                    .sum()
                    .rename(columns={
                        cu_selected: "__CAT__",
                        budget_acct_col: "__ACCT__",
                        route_val_col: "BudgetValue"
                    })
                )

                if pair.empty:
                    st.info("No budget data available to determine category→account mapping.")
                else:
                    pair_sorted = pair.sort_values(["__CAT__", "BudgetValue"], ascending=[True, False])
                    dominant = pair_sorted.drop_duplicates(subset=["__CAT__"], keep="first")[["__CAT__", "__ACCT__"]]
                    dominant = dominant.rename(columns={"__CAT__": "Category", "__ACCT__": "Account"})

                    routed = actuals_cat.merge(dominant, on="Category", how="left")
                    routed["Account"] = routed["Account"].fillna("Unmapped")

                    payments = (
                        routed.groupby("Account", as_index=False)["Actuals"]
                        .sum()
                        .rename(columns={"Actuals": "To Pay"})
                    )
                    if hide_zero_payments:
                        payments = payments[payments["To Pay"] != 0]

                    if not payments.empty:
                        payments = payments.sort_values("To Pay", ascending=False)

                    st.dataframe(payments, use_container_width=True)

                    with st.expander("See routing details (Category → Account → Actuals)", expanded=False):
                        detail = routed.rename(columns={"Actuals": "Amount"})
                        if hide_zero_payments:
                            detail = detail[detail["Amount"] != 0]
                        detail = detail.sort_values(["Account", "Amount"], ascending=[True, False])
                        st.dataframe(detail, use_container_width=True)

                    dl = payments.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download payments CSV",
                        data=dl,
                        file_name=f"payments_{locals().get('period_selected','period')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

with tab_settings:
    # =========================
    # Settings: Manage saved transactions (moved here)
    # =========================
    section_header("🧹", "Manage saved transactions")
    if not ws_id:
        st.info("Enter a Workspace ID to manage saved transactions.")
    else:
        with st.expander("Delete ALL transactions (workspace-wide)", expanded=False):
            st.warning(
                "This will permanently remove **all** saved transactions for this workspace. "
                "Consider downloading a backup first."
            )

            try:
                _existing_tx = read_transactions(ws_id)
                if _existing_tx is not None and not _existing_tx.empty:
                    _csv_bak = _existing_tx.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download current transactions (CSV backup)",
                        data=_csv_bak,
                        file_name=f"transactions_backup_{ws_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.caption("No saved transactions found; nothing to back up.")
            except Exception as _e:
                st.caption(f"Could not read existing transactions for backup: {_e}")

            col_dz1, col_dz2 = st.columns([1, 2])
            with col_dz1:
                sure_del = st.checkbox("I'm sure", value=False, key="confirm_delete_all_tx")
            with col_dz2:
                if st.button("Delete ALL transactions", type="secondary", disabled=not sure_del, use_container_width=True):
                    try:
                        try:
                            _existing_tx2 = read_transactions(ws_id)
                            if _existing_tx2 is not None and not _existing_tx2.empty:
                                empty_schema = list(_existing_tx2.columns)
                            else:
                                empty_schema = [
                                    "Date","Description","Merchant","Account","Amount",
                                    "AssignedCategory","Notes","Period","TX_ID"
                                ]
                        except Exception:
                            empty_schema = [
                                "Date","Description","Merchant","Account","Amount",
                                "AssignedCategory","Notes","Period","TX_ID"
                            ]

                        write_transactions(ws_id, pd.DataFrame(columns=empty_schema))
                        st.session_state.pop("tx_staged", None)
                        st.session_state.pop("tx_universe", None)
                        st.session_state["ws_tx_df"] = pd.DataFrame()  # clear snapshot

                        st.success("All transactions were deleted for this workspace.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
