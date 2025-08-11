# app/streamlit_app.py (imports)
import io
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts, JsCode

from utils.echarts import make_echarts_sankey_options
from utils.io import read_csv_any, read_table_any, get_excel_sheets, validate_upload
from utils.sankey import prepare_sankey_data, make_sankey_figure
from utils.constants import SAMPLE_CSV, ALLOWED_EXTS, ALLOWED_MIME, MAX_UPLOAD_MB


st.set_page_config(page_title="Budget App", layout="wide")
st.title("Budget App")

# --- Sidebar: Upload + Options ---
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

# Load data (uploaded or sample)
if uploaded is not None:
    ok, msg = validate_upload(
        uploaded,
        allowed_exts=ALLOWED_EXTS,
        allowed_mime=ALLOWED_MIME,
        max_bytes=MAX_UPLOAD_MB * 1024 * 1024,
    )
    if not ok:
        st.error(msg)
        st.stop()  # prevent fallback to sample; force user to upload a valid file

    source_name = uploaded.name
    ext = source_name.split(".")[-1].lower()
    if ext in {"xlsx", "xls", "xlsm"}:
        sheets = get_excel_sheets(uploaded)
        sheet = st.sidebar.selectbox("Sheet", sheets, index=0)
        df_raw = read_table_any(uploaded, sheet_name=sheet)
    else:
        df_raw = read_table_any(uploaded)
else:
    st.info("No file uploaded. Using the built-in **sample dataset**.")
    df_raw = read_csv_any(io.StringIO(SAMPLE_CSV))
    source_name = "sample.csv"


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

# --------------------------
# Chart style select
# --------------------------

st.sidebar.header("3) Visualization")
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

# --------------------------
# Build & Render Sankey
# --------------------------
st.subheader("Income flow")
try:
    sankey_data = prepare_sankey_data(df_in, l1=l1, l2=l2, l3=l3, value_col=val_col)

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