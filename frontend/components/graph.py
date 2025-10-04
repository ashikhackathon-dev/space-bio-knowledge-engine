# frontend/components/graph.py
import dash_cytoscape as cyto
from dash import html

cyto.load_extra_layouts()

DEFAULT_STYLESHEET = [
    {"selector": "node", "style": {"label": "data(label)", "font-size": "10px", "color": "var(--text)" }},
    {"selector": 'node[type = "Publication"]', "style": {"shape": "rectangle", "background-color": "#5C9DED"}},
    {"selector": 'node[type = "Experiment"]', "style": {"shape": "round-rectangle", "background-color": "#78D3F8"}},
    {"selector": 'node[type = "Organism"]', "style": {"shape": "ellipse", "background-color": "#38D996"}},
    {"selector": 'node[type = "Assay"]', "style": {"shape": "diamond", "background-color": "#B07CFF"}},
    {"selector": 'node[type = "Mission"]', "style": {"shape": "hexagon", "background-color": "#5CA0FF"}},
    {"selector": 'node[type = "Tissue"]', "style": {"shape": "tag", "background-color": "#F1C40F"}},
    {"selector": 'node[type = "Gene"]', "style": {"shape": "triangle", "background-color": "#FF9FF3"}},
    {"selector": 'node[type = "Protein"]', "style": {"shape": "vee", "background-color": "#48C9B0"}},
    {"selector": 'node[type = "Pathway"]', "style": {"shape": "rhomboid", "background-color": "#F39C12"}},
    {"selector": 'node[type = "Hazard"]', "style": {"shape": "octagon", "background-color": "#FF6B6B"}},
    {"selector": 'node[type = "Countermeasure"]', "style": {"shape": "vee", "background-color": "#16A085"}},
    {"selector": "edge", "style": {"curve-style": "bezier", "width": 1.5, "label": "data(label)"}}
]

def build_graph(elements=None, layout="cose", height="600px", stylesheet=None, id="knowledge-graph"):
    elements = elements or []
    stylesheet = stylesheet or DEFAULT_STYLESHEET
    return cyto.Cytoscape(
        id=id,
        elements=elements,
        layout={"name": layout},
        zoom=1, minZoom=0.1, maxZoom=2,
        pan={"x":0,"y":0}, boxSelectionEnabled=True,
        style={"width": "100%", "height": height, "backgroundColor": "var(--bg)"},
        stylesheet=stylesheet
    )

