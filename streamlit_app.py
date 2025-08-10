import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Budget App", layout="wide")
st.title("Budget Visualization App")

# --------------------------
# Helpers
# --------------------------
SAMPLE_CSV = """Category,Subcategory,Bank Account,Amount
Income 1,Internet/Phone/Insurance,Bank3,250
Income 1,Travel,Bank1,413
Income 1,Spending,Bank1,900
Income 1,Student Loans,Bank2,82.7
Income 1,Personal,Bank2,167.31
Income 1,Saving Das,Bank1,200
Income 1,Rent,Bank1,1300
Income 1,Extra,Bank1,450
Income 2,Mortgage,Bank2,1726.29
Income 2,PM,Bank2,225
Income 2,Pest control,Bank2,69
Income 2,house Repairs,Bank2,199.71
Income 2,Lawn,Bank2,225
"""

def read_csv_any(file_or_buffer, **kwargs):
    """Read CSV from upload or bytes, with a few sensible defaults."""
    try:
        return pd.read_csv(file_or_buffer, **kwargs)
    except Exception:
        # Try semicolon delimiter and comma decimal as fallback
        file_or_buffer.seek(0) if hasattr(file_or_buffer, "seek") else None
        return pd.read_csv(file_or_buffer, sep=";", decimal=",", **kwargs)

def prepare_sankey_data(df, l1, l2, l3, value_col):
    # Validate columns
    missing = [c for c in [l1, l2, l3, value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Normalize types
    for col in [l1, l2, l3]:
        df[col] = df[col].astype(str).str.strip().fillna("Unknown")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Aggregate to avoid duplicate links
    g12 = df.groupby([l1, l2], as_index=False)[value_col].sum()
    g23 = df.groupby([l2, l3], as_index=False)[value_col].sum()

    # Build node list across all three levels, preserving order and uniqueness
    nodes_l1 = g12[l1].unique().tolist()
    nodes_l2 = g12[l2].unique().tolist()
    nodes_l3 = g23[l3].unique().tolist()

    nodes = nodes_l1 + [n for n in nodes_l2 if n not in nodes_l1] + \
            [n for n in nodes_l3 if n not in nodes_l1 and n not in nodes_l2]

    node_index = {name: i for i, name in enumerate(nodes)}

    # Links: L1 → L2
    src_12 = g12[l1].map(node_index).tolist()
    tgt_12 = g12[l2].map(node_index).tolist()
    val_12 = g12[value_col].tolist()

    # Links: L2 → L3
    src_23 = g23[l2].map(node_index).tolist()
    tgt_23 = g23[l3].map(node_index).tolist()
    val_23 = g23[value_col].tolist()

    sources = src_12 + src_23
    targets = tgt_12 + tgt_23
    values  = val_12 + val_23

    return {"nodes": nodes, "sources": sources, "targets": targets, "values": values}

def render_sankey(sankey_data, title="Three-Level Sankey"):
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['nodes'],
            hovertemplate="%{label}<extra></extra>"
        ),
        link=dict(
            source=sankey_data['sources'],
            target=sankey_data['targets'],
            value=sankey_data['values'],
            hovertemplate="Flow: %{value}<extra></extra>"
        )
    )])
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Sidebar: Upload + Options
# --------------------------
st.sidebar.header("1) Upload CSV")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.caption("No file? Download a sample to try:")
st.sidebar.download_button(
    "Download sample.csv",
    data=SAMPLE_CSV,
    file_name="sample.csv",
    mime="text/csv",
    use_container_width=True
)

# Load data (uploaded or sample)
if uploaded is not None:
    df_raw = read_csv_any(uploaded)
    source_name = uploaded.name
else:
    st.info("No file uploaded. Using the built-in **sample dataset**.")
    df_raw = read_csv_any(io.StringIO(SAMPLE_CSV))
    source_name = "sample.csv"

# --------------------------
# Column Mapping
# --------------------------
st.sidebar.header("2) Map Columns")
all_cols = list(df_raw.columns)

# Heuristics to preselect likely columns
def guess(col_names, candidates):
    for c in candidates:
        if c in col_names:
            return c
    return col_names[0] if col_names else None

l1 = st.sidebar.selectbox("Level 1 (e.g., Category)", all_cols, index=all_cols.index(guess(all_cols, ["Category"])) if all_cols else 0)
l2 = st.sidebar.selectbox("Level 2 (e.g., Subcategory)", all_cols, index=all_cols.index(guess(all_cols, ["Subcategory"])) if all_cols else 0)
l3 = st.sidebar.selectbox("Level 3 (e.g., Bank Account)", all_cols, index=all_cols.index(guess(all_cols, ["Bank Account","Account","Destination"])) if all_cols else 0)
val_col = st.sidebar.selectbox("Value column (numeric)", all_cols, index=all_cols.index(guess(all_cols, ["Amount","Value","Count"])) if all_cols else 0)

# Warn if duplicates in mapping
if len({l1, l2, l3}) < 3:
    st.sidebar.warning("Each level should be a different column.")

st.subheader("Data Preview")
st.caption(f"Source: **{source_name}**")
st.dataframe(df_raw.head(50), use_container_width=True)

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
st.subheader("Sankey Chart (3 levels)")
try:
    sankey_data = prepare_sankey_data(df_in, l1=l1, l2=l2, l3=l3, value_col=val_col)
    render_sankey(sankey_data, title=f"{l1} → {l2} → {l3}")
except Exception as e:
    st.error(f"Could not render Sankey chart: {e}")