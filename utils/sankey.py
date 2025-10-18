from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go

def prepare_sankey_data(
        df: pd.DataFrame,
        l1: str,
        l2: str,
        l3: str,
        value_col: str,
        l4: Optional[str] = None,
        ) -> Dict[str, List]:
    
    # Validate
    required = [l1, l2, l3, value_col]
    if l4:
        required.append(l4)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Normalize
    norm_cols = [l1, l2, l3] + ([l4] if l4 else [])
    for col in norm_cols:
        df[col] = df[col].astype(str).str.strip().fillna("Unknown")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Aggregate to avoid duplicate links
    g12 = df.groupby([l1, l2], as_index=False)[value_col].sum()
    g23 = df.groupby([l2, l3], as_index=False)[value_col].sum()
    g34 = None
    if l4:
        g34 = df.groupby([l3, l4], as_index=False)[value_col].sum()

    # Build node list across levels, preserving order and uniqueness
    nodes_l1 = g12[l1].unique().tolist()
    nodes_l2 = g12[l2].unique().tolist()
    nodes_l3 = g23[l3].unique().tolist()
    nodes_l4 = g34[l4].unique().tolist() if g34 is not None else []

    nodes = (
        nodes_l1
        + [n for n in nodes_l2 if n not in nodes_l1]
        + [n for n in nodes_l3 if n not in nodes_l1 and n not in nodes_l2]
        + [n for n in nodes_l4 if n not in nodes_l1 and n not in nodes_l2 and n not in nodes_l3]
        )

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
    
    # Optional Links: L3 → L4
    if g34 is not None:
        src_34 = g34[l3].map(node_index).tolist()
        tgt_34 = g34[l4].map(node_index).tolist()
        val_34 = g34[value_col].tolist()
        sources += src_34
        targets += tgt_34
        values  += val_34

    return {"nodes": nodes, "sources": sources, "targets": targets, "values": values}

def make_sankey_figure(sankey_data: Dict[str, List], title: str = "Sankey") -> go.Figure:
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
    return fig