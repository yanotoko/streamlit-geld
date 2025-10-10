# app/streamlit_app.py (imports)
import io
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from utils.echarts import make_echarts_sankey_options
from utils.io import read_csv_any, read_table_any, get_excel_sheets, validate_upload
from utils.sankey import prepare_sankey_data, make_sankey_figure

from utils.constants import (
    SAMPLE_CSV, DEFAULT_HEADERS, ALLOWED_EXTS, ALLOWED_MIME, MAX_UPLOAD_MB,
    LOOKUP_HEADERS, FREQUENCY_CHOICES,
)
from utils.sheets import (
    read_workspace, write_workspace, sanitize_workspace_id, VersionConflict,
    read_lookup_for_ws, write_lookup_for_ws,
)


# Streamlit rerun compatibility
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


st.set_page_config(page_title="Budget App", layout="wide")
st.title("Budget App")

# --- Sidebar: Upload + Options ---
# --- Sidebar: Workspace (public app uses Workspace ID) ---
st.sidebar.header("0) Workspace")
ws_input = st.sidebar.text_input("Workspace ID", value=st.session_state.get("workspace_id", ""))
col_ws1, col_ws2 = st.sidebar.columns([1, 1])
with col_ws1:
    load_ws = st.button("Load workspace", use_container_width=True)
with col_ws2:
    # optional UX: create if missing (load does this anyway); kept for clarity
    create_ws = st.button("Create if missing", use_container_width=True)

if load_ws or create_ws:
    st.session_state["workspace_id"] = ws_input.strip()


st.sidebar.header("1) Upload file")
uploaded = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTS]
)
st.sidebar.caption(f"Allowed: CSV, XLSX/XLS/XLSM • Max size: {MAX_UPLOAD_MB} MB")

st.sidebar.download_button(
    "Download sample.csv",
    data=SAMPLE_CSV,
    file_name="sample.csv",
    mime="text/csv",
    use_container_width=True
)

# 3c — Load order with seeding/import support
df_raw = None
source_name = ""
df_uploaded = None

ws_id = st.session_state.get("workspace_id", "").strip()
ws_tab = sanitize_workspace_id(ws_id) if ws_id else ""
ws_loaded = False
ws_empty = True

# Try to read workspace (but don't lock in as df_raw yet)
if ws_id:
    try:
        df_ws, meta = read_workspace(ws_id, DEFAULT_HEADERS)
        ws_loaded = True
        ws_empty = df_ws.dropna(how="all").shape[0] == 0
        st.session_state["ws_version"] = int(meta.get("version", 0))
    except Exception as e:
        st.error(f"Could not load workspace '{ws_id}': {e}")

# Parse uploaded file regardless of workspace state
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
        sheet = st.sidebar.selectbox("Sheet", sheets, index=0)
        df_uploaded = read_table_any(uploaded, sheet_name=sheet)
    else:
        df_uploaded = read_table_any(uploaded)

# Decide what to show/edit
if ws_loaded and not ws_empty:
    df_raw = df_ws
    source_name = f"Google Sheets (tab: {ws_tab})"
elif df_uploaded is not None:
    df_raw = df_uploaded
    source_name = upload_name
else:
    st.info("No workspace data or upload found. Using the built-in **sample dataset**.")
    df_raw = read_csv_any(io.StringIO(SAMPLE_CSV))
    source_name = "sample.csv"

# If both exist (non-empty workspace AND an upload), let user pick
if ws_loaded and not ws_empty and df_uploaded is not None:
    choice = st.sidebar.radio("Use data from", ("Workspace", "Uploaded file"), index=0)
    if choice == "Uploaded file":
        df_raw = df_uploaded
        source_name = upload_name

# 3e — Import uploaded file into the selected workspace
if ws_id and df_uploaded is not None:
    st.sidebar.markdown("---")
    if st.sidebar.button("Import uploaded → Workspace", use_container_width=True):
        try:
            expected = None if ws_empty else st.session_state.get("ws_version", 0)
            new_version = write_workspace(ws_id, df_uploaded, expected_version=expected)
            st.session_state["ws_version"] = new_version
            st.sidebar.success(f"Imported into '{ws_id}' (version {new_version}).")
            st.rerun()
        except VersionConflict as vc:
            st.sidebar.warning(str(vc))
            col_imp1, col_imp2 = st.sidebar.columns(2)
            with col_imp1:
                if st.button("Reload workspace"):
                    df_ws, meta = read_workspace(ws_id, DEFAULT_HEADERS)
                    st.session_state["ws_version"] = int(meta.get("version", 0))
                    st.rerun()
            with col_imp2:
                if st.button("Force overwrite"):
                    try:
                        new_version = write_workspace(ws_id, df_uploaded, expected_version=None)
                        st.session_state["ws_version"] = new_version
                        st.sidebar.success(f"Overwrote '{ws_id}' (version {new_version}).")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Overwrite failed: {e}")
        except Exception as e:
            st.sidebar.error(f"Import failed: {e}")


# --------------------------
# Column Mapping
# --------------------------
st.sidebar.header("2) Map Columns")
all_cols = list(df_raw.columns)

def guess(col_names, candidates):
    for c in candidates:
        if c in col_names:
            return c
    return col_names[0] if col_names else None

l1 = st.sidebar.selectbox("Level 1 (e.g., Income)", all_cols, index=all_cols.index(guess(all_cols, ["Income", "Income"])) if all_cols else 0)
l2 = st.sidebar.selectbox("Level 2 (e.g., Category)", all_cols, index=all_cols.index(guess(all_cols, ["Subcategory", "Category"])) if all_cols else 0)
l3 = st.sidebar.selectbox("Level 3 (e.g., Bank Account)", all_cols, index=all_cols.index(guess(all_cols, ["Bank Account","Account","Destination"])) if all_cols else 0)
val_col = st.sidebar.selectbox("Value column (numeric)", all_cols, index=all_cols.index(guess(all_cols, ["Amount","Value","Count"])) if all_cols else 0)

if len({l1, l2, l3}) < 3:
    st.sidebar.warning("Each level should be a different column.")

st.subheader("Data Preview")
st.caption(f"Source: **{source_name}**")
st.dataframe(df_raw.head(50), use_container_width=True)

# --- Load per-workspace lookup (seeded from current Level1 values if new/empty) ---
ws_id = st.session_state.get("workspace_id", "").strip()

if ws_id:
    # Build a starter from current data's unique Level1 values (default monthly, factor 1.0)
    starter = None
    if l1 in df_raw.columns:
        starter_levels = sorted(
            df_raw[l1].dropna().astype(str).str.strip().unique().tolist()
        )
        if starter_levels:
            starter = [{"Level1": x, "Frequency": "monthly", "Factor_per_month": 1.0} for x in starter_levels]

    try:
        df_lookup = read_lookup_for_ws(ws_id, starter_rows=starter)
    except Exception as e:
        st.warning(f"Could not load lookup for '{ws_id}': {e}")
        df_lookup = pd.DataFrame(columns=LOOKUP_HEADERS)
else:
    st.info("No workspace selected — per-workspace lookup disabled.")
    df_lookup = pd.DataFrame(columns=LOOKUP_HEADERS)


# Edit Lookup frequency values
st.sidebar.header("2b) Frequency Lookup (per workspace)")
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
            st.rerun()
        except Exception as e:
            st.error(f"Saving lookup failed: {e}")


# 2c) Filter (applies ONLY to Level 1)
st.sidebar.header("2c) Filter")
level1_values_all = sorted([str(x) for x in df_raw[l1].dropna().unique().tolist()]) if l1 in df_raw.columns else []
level1_selected = st.sidebar.multiselect("Show only these Level 1 values", options=level1_values_all, default=level1_values_all)


# --------------------------
# Chart style select
# --------------------------

st.sidebar.header("3) Visualization")
normalize = st.sidebar.checkbox("Normalize to monthly equivalents (uses lookup factors)", value=False)

chart_type = st.sidebar.selectbox(
    "Choose a chart",
    ["Sankey (Plotly)", "Sankey (ECharts, labels always on)"],
    index=1
)

# Optional ECharts-only knobs
if chart_type == "Sankey (ECharts, labels always on)":
    ech_curveness = st.sidebar.slider("ECharts link curveness", 0.0, 1.0, 0.5, 0.05)
    ech_edge_font = st.sidebar.number_input("Edge label font size", 8, 24, 11, 1)


# --------------------------
# Editable table
# --------------------------
st.subheader("Edit Data")
edited_df = st.data_editor(
    df_raw,
    num_rows="dynamic",
    use_container_width=True,
    key="editor"
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Use Edited Data"):
        df_in = edited_df.copy()
        st.success("Using edited data for the chart.")
    else:
        df_in = df_raw.copy()

with col_b:
    csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Edited CSV", csv_bytes, "edited_data.csv", "text/csv", use_container_width=True)


viz_df = edited_df.copy()

# Apply Level-1 filter (visualization only)
if l1 in viz_df.columns and 'level1_selected' in locals() and level1_selected:
    viz_df = viz_df[viz_df[l1].astype(str).isin(level1_selected)]

# Decide which value column to feed the Sankey
val_to_use = val_col
if ws_id and not df_lookup.empty and l1 in viz_df.columns and normalize:
    df_norm = viz_df.merge(
        df_lookup[["Level1", "Factor_per_month"]],
        how="left",
        left_on=l1,
        right_on="Level1"
    )
    df_norm["Factor_per_month"] = pd.to_numeric(df_norm["Factor_per_month"], errors="coerce").fillna(1.0)
    df_norm["_AmountMonthly"] = pd.to_numeric(df_norm[val_col], errors="coerce").fillna(0) * df_norm["Factor_per_month"]
    val_to_use = "_AmountMonthly"
else:
    df_norm = viz_df.copy()


# 3d — Save to Google Sheets workspace
st.divider()
save_col1, save_col2 = st.columns([1, 3])
with save_col1:
    can_save = bool(st.session_state.get("workspace_id"))
    save_btn = st.button("Save to Workspace", type="primary", disabled=not can_save)

if not can_save:
    st.caption("Enter a Workspace ID in the sidebar to enable saving.")

if save_btn and can_save:
    ws_id = st.session_state["workspace_id"]
    expected_version = st.session_state.get("ws_version", 0)
    df_to_save = edited_df.copy()   # always save what the user is seeing in the editor
    try:
        new_version = write_workspace(ws_id, df_to_save, expected_version=expected_version)
        st.session_state["ws_version"] = new_version
        st.success(f"Saved workspace '{ws_id}' (version {new_version}).")
        _rerun()
    except VersionConflict as vc:
        st.warning(str(vc))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reload from Sheets"):
                df_ws, meta = read_workspace(ws_id, DEFAULT_HEADERS)
                st.session_state["ws_version"] = int(meta.get("version", 0))
                st.rerun()
        with c2:
            if st.button("Overwrite anyway"):
                try:
                    new_version = write_workspace(ws_id, df_to_save, expected_version=None)
                    st.session_state["ws_version"] = new_version
                    st.success(f"Overwrote workspace '{ws_id}' (version {new_version}).")
                except Exception as e:
                    st.error(f"Could not overwrite: {e}")
    except Exception as e:
        st.error(f"Save failed: {e}")


# Build a visualization DataFrame from the editor (not saved data)
viz_df = edited_df.copy()

# Apply Level-1 filter (visualization only)
if l1 in viz_df.columns and level1_selected:
    viz_df = viz_df[viz_df[l1].astype(str).isin(level1_selected)]


# --------------------------
# Build & Render Sankey
# --------------------------

# Decide which value column to feed the Sankey
val_to_use = val_col
if normalize and l1 in viz_df.columns:
    # join to lookup on Level1 (left_on=l1, right_on='Level1')
    join_cols = ["Level1", "Factor_per_month", "Frequency"]
    df_norm = viz_df.merge(df_lookup[["Level1", "Factor_per_month"]], how="left",
                           left_on=l1, right_on="Level1")
    df_norm["Factor_per_month"] = pd.to_numeric(df_norm["Factor_per_month"], errors="coerce").fillna(1.0)
    df_norm["_AmountMonthly"] = pd.to_numeric(df_norm[val_col], errors="coerce").fillna(0) * df_norm["Factor_per_month"]
    val_to_use = "_AmountMonthly"
else:
    df_norm = viz_df.copy()

st.subheader("Income flow")
try:
    #sankey_data = prepare_sankey_data(df_in, l1=l1, l2=l2, l3=l3, value_col=val_col)
    sankey_data = prepare_sankey_data(df_norm, l1=l1, l2=l2, l3=l3, value_col=val_to_use)


    if chart_type == "Sankey (Plotly)":
        fig = make_sankey_figure(
            sankey_data,
            title=f"{l1} → {l2} → {l3}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Sankey (ECharts, labels always on)":
        options = make_echarts_sankey_options(
            sankey_data,
            title=f"{l1} → {l2} → {l3}",
            value_suffix=" $",        # optional
            curveness=ech_curveness,
            edge_font_size=int(ech_edge_font),
            py_value_format=",.0f",   # ",.2f" for cents, etc.
        )
        st_echarts(options=options, height="600px")

    else:
        st.warning(f"Unsupported chart type: {chart_type}")

except Exception as e:
    st.error(f"Could not render chart: {e}")


# st.subheader("Income flow")
# try:
#     sankey_data = prepare_sankey_data(df_in, l1=l1, l2=l2, l3=l3, value_col=val_col)
#     fig = make_sankey_figure(sankey_data, title=f"{l1} → {l2} → {l3}")
#     st.plotly_chart(fig, use_container_width=True)
# except Exception as e:
#     st.error(f"Could not render Sankey chart: {e}")