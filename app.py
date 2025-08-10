import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'budget.csv')

st.set_page_config(page_title="Budget Visualization App", layout="wide")

st.title("Budget Visualization App")

# Load budget data
@st.cache_data
def load_budget(path):
    return pd.read_csv(path)

def save_budget(df, path):
    df.to_csv(path, index=False)

def prepare_sankey_data(df):
    # Example: assumes columns 'Category', 'Subcategory', 'Amount'
    nodes = list(pd.unique(df['Category'].tolist() + df['Subcategory'].tolist()))
    node_indices = {name: i for i, name in enumerate(nodes)}
    sources = df['Category'].map(node_indices)
    targets = df['Subcategory'].map(node_indices)
    values = df['Amount']
    return {
        "nodes": nodes,
        "sources": sources,
        "targets": targets,
        "values": values
    }

# Load data
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at {DATA_PATH}")
    st.stop()

df = load_budget(DATA_PATH)

st.subheader("Budget Table (Editable)")

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="budget_editor"
)

if st.button("Save Changes"):
    save_budget(edited_df, DATA_PATH)
    st.success("Changes saved to CSV.")

# Add new row functionality
st.markdown("### Add New Row")
with st.form("add_row_form", clear_on_submit=True):
    col1, col2, col3, col4 = st.columns(4)
    new_category = col1.text_input("Category")
    new_subcategory = col2.text_input("Subcategory")
    new_amount = col3.number_input("Amount", value=0.0, step=0.01)
    new_account = col4.text_input("Bank Account")
    submitted = st.form_submit_button("Add Row")
    if submitted:
        new_row = {
            "Category": new_category,
            "Subcategory": new_subcategory,
            "Amount": new_amount,
            "Bank Account": new_account
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_budget(df, DATA_PATH)
        st.success("New row added. Please refresh the page to see the update.")

# Sankey chart
st.subheader("Sankey Chart")
try:
    sankey_data = prepare_sankey_data(df)
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['nodes']
        ),
        link=dict(
            source=sankey_data['sources'],
            target=sankey_data['targets'],
            value=sankey_data['values']
        )
    )])
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render Sankey chart: {e}")