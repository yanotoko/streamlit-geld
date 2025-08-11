# utils/echarts.py
from __future__ import annotations
from typing import Dict, List

def make_echarts_sankey_options(
    sankey_data: Dict[str, List],
    title: str,
    value_suffix: str = " $",
    curveness: float = 0.5,
    edge_font_size: int = 11,
    py_value_format: str = ",.0f",   # Python-side number formatting
):
    names = sankey_data["nodes"]
    sources = sankey_data["sources"]
    targets = sankey_data["targets"]
    values  = sankey_data["values"]

    nodes = [{"name": n} for n in names]

    def fmt(v: float) -> str:
        try:
            return f"{float(v):{py_value_format}}{value_suffix}"
        except Exception:
            return f"{v}{value_suffix}"

    # Preformat label into link.name; keep numeric value for thickness
    links = [
        {
            "source": names[s],
            "target": names[t],
            "value": float(v),
            "name": fmt(v),   # <- label text lives here
        }
        for s, t, v in zip(sources, targets, values)
    ]

    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"show": True, "trigger": "item", "triggerOn": "mousemove|click"},
        "series": [{
            "type": "sankey",
            "data": nodes,
            "links": links,
            "lineStyle": {"color": "source", "curveness": curveness},
            "label": {"show": True},  # node labels
            "edgeLabel": {
                "show": True,
                "formatter": "{b}",    # <- use link.name
                "position": "middle",
                "fontSize": edge_font_size,
                "color": "#000"
            }
        }]
    }
