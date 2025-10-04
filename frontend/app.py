# frontend/app.py
import os
import json
import time
from urllib.parse import urlencode, urlparse, parse_qs

import requests
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, dcc, html, Input, Output, State, ctx, MATCH, ALL

# Load extra layouts for Cytoscape
cyto.load_extra_layouts()

# Load environment variables
load_dotenv()

# Backend base URL (Flask default 5000)
BACKEND_BASE = os.environ.get("FRONTEND_BACKEND_URL", "http://localhost:5000")

# Global backend availability flag (updated at startup)
backend_available = False

# Try connecting to backend
def initial_backend_check():
    candidates = [f"{BACKEND_BASE}/health", f"{BACKEND_BASE}/api/health", f"{BACKEND_BASE}/"]
    for url in candidates:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False

backend_available = initial_backend_check()

# App initialization with dark bootstrap theme (CYBORG by default)
THEME = dbc.themes.CYBORG  # swap to dbc.themes.DARKLY if preferred
app = Dash(
    __name__,
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True,
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Space Biology Knowledge Engine | NASA OSDR Explorer"
server = app.server  # for WSGI if needed

# Helpers
def api_get(path, params=None, timeout=10):
    url = f"{BACKEND_BASE}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"GET {path} failed: {e}")

def api_post(path, json_body=None, timeout=30):
    url = f"{BACKEND_BASE}{path}"
    try:
        r = requests.post(url, json=json_body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"POST {path} failed: {e}")

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def short(text, n=220):
    if not text:
        return ""
    s = str(text).strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def parse_url_query(query_string):
    if not query_string:
        return {}
    q = parse_qs(query_string.lstrip("?"))
    state = {}
    for k, v in q.items():
        # store single values as str, lists as list
        if len(v) == 1:
            state[k] = v[0]
        else:
            state[k] = v
    # JSON decode complex fields if present
    for key in ["filters", "selected_ids"]:
        if key in state:
            try:
                state[key] = json.loads(state[key])
            except Exception:
                pass
    return state

def build_url_query(state_dict):
    enc = {}
    for k, v in state_dict.items():
        if isinstance(v, (dict, list)):
            enc[k] = json.dumps(v, separators=(",", ":"))
        else:
            enc[k] = v
    return "?" + urlencode(enc, doseq=False)

# Cytoscape styles
CY_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": "10px",
            "text-valign": "center",
            "text-halign": "center",
            "color": "var(--text)",
            "background-color": "#666",
            "border-color": "#999",
            "border-width": 1,
            "text-outline-color": "rgba(0,0,0,0.35)",
            "text-outline-width": 2,
        },
    },
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
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "width": 1.5,
            "line-color": "#888",
            "target-arrow-color": "#aaa",
            "target-arrow-shape": "vee",
            "opacity": 0.9,
            "label": "data(label)",
            "font-size": "9px",
            "text-background-color": "rgba(0,0,0,0.3)",
            "text-background-opacity": 1,
            "text-background-shape": "round-rectangle",
        },
    },
]

def to_cyto_elements(graph_data):
    if not graph_data:
        return []
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    cy_nodes = [{"data": {"id": n.get("id"), "label": n.get("label"), "type": n.get("type"), "meta": n.get("meta", {})}} for n in nodes if n.get("id")]
    cy_edges = [{"data": {"source": e.get("source"), "target": e.get("target"), "label": e.get("label"), "rel_type": e.get("rel_type")}} for e in edges if e.get("source") and e.get("target")]
    return cy_nodes + cy_edges

def render_result_card(item):
    item_id = item.get("id") or item.get("object_id") or item.get("publication_id") or item.get("experiment_id") or ""
    title = item.get("title") or item.get("publication_title") or f"Item {item_id}"
    snippet = item.get("snippet") or short(item.get("abstract") or item.get("chunk_text") or "")
    organism = item.get("organism")
    mission = item.get("mission")
    assay = item.get("assay")
    accession = item.get("study_accession")
    score = item.get("score")
    badges = []
    if accession:
        badges.append(dbc.Badge(accession, color="info", className="me-1"))
    if organism:
        badges.append(dbc.Badge(organism, color="success", className="me-1"))
    if mission:
        badges.append(dbc.Badge(mission, color="primary", className="me-1"))
    if assay:
        badges.append(dbc.Badge(assay, color="secondary", className="me-1"))
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                dbc.Button(
                    title,
                    id={"type": "result-select", "id": str(item_id)},
                    color="link",
                    className="p-0 result-title-btn",
                ),
                className="d-flex align-items-start justify-content-between",
            ),
            html.Div(badges, className="mb-1"),
            html.P(snippet, className="text-truncate-2 mb-1"),
            html.Div([
                dbc.Badge(f"{score:.2f}" if isinstance(score, (float, int)) else "score", color="dark", className="me-2"),
                dcc.Checklist(
                    id={"type": "result-check", "id": str(item_id)},
                    options=[{"label": " select", "value": str(item_id)}],
                    value=[],
                    inputStyle={"marginRight": "6px"}
                ),
            ], className="d-flex align-items-center"),
        ]),
        className="mb-2 result-card",
    )

def aggregate_for_analytics(results):
    if not results:
        return {"organism": {}, "assay": {}, "mission": {}, "tissue": {}, "timeline": {}}
    df = pd.DataFrame(results)
    agg = {}
    for col in ["organism", "assay", "mission", "tissue"]:
        if col in df.columns:
            counts = df[col].dropna().astype(str).value_counts().to_dict()
        else:
            counts = {}
        agg[col] = counts
    # timeline by year
    year_counts = {}
    if "year" in df.columns:
        y = df["year"].dropna()
        try:
            y = y.astype(int)
            year_counts = y.value_counts().sort_index().to_dict()
        except Exception:
            pass
    agg["timeline"] = year_counts
    return agg

def analytics_fig_from_counts(counts, title):
    if not counts:
        fig = px.bar(title=title)
        fig.update_layout(template="plotly_dark", annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        return fig
    df = pd.DataFrame({"label": list(counts.keys()), "value": list(counts.values())})
    fig = px.bar(df, x="label", y="value", title=title)
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

def timeline_fig_from_counts(year_counts):
    if not year_counts:
        fig = px.bar(title="Studies over time")
        fig.update_layout(template="plotly_dark", annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        return fig
    ys = sorted(year_counts.items(), key=lambda kv: kv[0])
    df = pd.DataFrame(ys, columns=["year", "count"])
    fig = px.bar(df, x="year", y="count", title="Studies over time")
    fig.update_layout(template="plotly_dark", xaxis=dict(type="category"))
    return fig

# Layout
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([
            html.Span("🚀", className="me-2"),
            html.Span("Space Biology Knowledge Engine | NASA OSDR Explorer")
        ], className="brand-text"),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Row([
            dbc.Col(dbc.Input(id="query-input", type="search", debounce=True,
                              placeholder="Search space biology studies...", className="me-2", size="md"), width="auto"),
            dbc.Col(dbc.Button("Search", id="search-btn", color="primary", n_clicks=0, className="me-2"), width="auto"),
            dbc.Col(dbc.Button("Filters", id="filters-toggle", color="secondary", outline=True, n_clicks=0), width="auto"),
        ], align="center", className="g-2"),
    ], fluid=True),
    color="dark", dark=True, className="mb-0"
)

backend_alert = html.Div(
    dbc.Alert(
        f"Backend unreachable at {BACKEND_BASE}. Some features are disabled.",
        color="warning", className="mb-0", id="backend-alert", is_open=not backend_available
    )
)

filters_panel = dbc.Collapse(
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="filter-organism", multi=True, placeholder="All organisms"), md=3),
                dbc.Col(dcc.Dropdown(id="filter-assay", multi=True, placeholder="All assays"), md=3),
                dbc.Col(dcc.Dropdown(id="filter-mission", multi=True, placeholder="All missions"), md=3),
                dbc.Col(dcc.Dropdown(id="filter-tissue", multi=True, placeholder="All tissues"), md=3),
            ], className="mb-3 g-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Year range", className="mb-2"),
                    dcc.RangeSlider(
                        id="year-slider", min=2000, max=2025, value=[2000, 2025],
                        marks=None, tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], md=12),
            ], className="mb-3"),
            html.Div([
                dbc.Button("Clear filters", id="clear-filters", color="light", outline=True, className="me-2"),
                dbc.Button("Apply", id="apply-filters", color="primary", className="me-2"),
            ]),
        ]),
        className="bg-dark border-0 shadow-sm"
    ),
    id="filters-collapse", is_open=False
)

results_panel = dbc.Card(
    [
        dbc.CardHeader(html.Div([
            html.Span("Search Results", className="me-2"),
            dbc.Badge("0", id="results-count-badge", color="info")
        ], className="d-flex align-items-center justify-content-between")),
        dbc.CardBody(
            dcc.Loading(
                id="loading-results",
                type="default",
                children=html.Div(id="results-list", style={"height": "70vh", "overflowY": "auto"})
            )
        )
    ],
    className="mb-3"
)

tabs_panel = dbc.Tabs(
    [
        dbc.Tab(
            dbc.Card(
                dbc.CardBody([
                    dbc.Input(id="question-input", placeholder="Ask a question (e.g., microgravity effects on bone?)", type="text", debounce=True, className="mb-2"),
                    dcc.RadioItems(
                        id="summary-style",
                        options=[{"label": s, "value": s} for s in ["abstract","bullet","methods","clinician"]],
                        value="abstract", inline=True, className="mb-2"
                    ),
                    dbc.Button("Summarize", id="summarize-btn", color="primary", className="mb-3"),
                    dcc.Loading(html.Pre(id="summary-output", className="summary-block"))
                ])
            ),
            label="Summary", tab_id="tab-summary"
        ),
        dbc.Tab(
            dbc.Card(
                dbc.CardBody([
                    dcc.Dropdown(
                        id="graph-layout",
                        options=[{"label": x, "value": x} for x in ["cose","cola","breadthfirst","concentric"]],
                        value="cose", clearable=False, style={"width":"220px"}, className="mb-2"
                    ),
                    dcc.Loading(
                        cyto.Cytoscape(
                            id="knowledge-graph",
                            elements=[],
                            layout={"name": "cose"},
                            style={"width":"100%","height":"600px","backgroundColor":"var(--bg)"},
                            stylesheet=CY_STYLESHEET,
                            zoom=1, minZoom=0.1, maxZoom=2,
                            pan={"x":0,"y":0}, boxSelectionEnabled=True
                        )
                    ),
                    html.Div(id="graph-node-details", className="mt-2")
                ])
            ),
            label="Knowledge Graph", tab_id="tab-graph"
        ),
        dbc.Tab(
            dbc.Card(
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="timeline-graph", figure=dict(data=[], layout=dict(template="plotly_dark"))))
                ])
            ),
            label="Timeline", tab_id="tab-timeline"
        ),
        dbc.Tab(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="organism-bar"), md=6),
                        dbc.Col(dcc.Graph(id="assay-bar"), md=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="mission-bar"), md=6),
                        dbc.Col(dcc.Graph(id="tissue-top"), md=6),
                    ])
                ])
            ),
            label="Analytics", tab_id="tab-analytics"
        ),
    ],
    active_tab="tab-summary",
)

app.layout = dbc.Container([
    navbar,
    backend_alert,
    filters_panel,
    dcc.Store(id="search-results", storage_type="memory"),
    dcc.Store(id="selected-ids", storage_type="memory"),
    dcc.Store(id="graph-data", storage_type="memory"),
    dcc.Store(id="filters-state", storage_type="memory"),
    dcc.Store(id="analytics-data", storage_type="memory"),
    dcc.Store(id="backend-status", storage_type="memory", data=backend_available),
    dcc.Location(id="url", refresh=False),
    dbc.Row([
        dbc.Col(results_panel, md=4),
        dbc.Col(tabs_panel, md=8),
    ], className="mt-3 g-3")
], fluid=True)

# Callbacks

@app.callback(
    Output("filters-collapse", "is_open"),
    Input("filters-toggle", "n_clicks"),
    State("filters-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_filters(n, is_open):
    return not is_open

@app.callback(
    Output("backend-status", "data"),
    Output("backend-alert", "is_open"),
    Input("url", "href"),
)
def on_load_backend_status(_):
    ok = initial_backend_check()
    return ok, (not ok)

@app.callback(
    Output("filters-state", "data"),
    Input("apply-filters", "n_clicks"),
    Input("clear-filters", "n_clicks"),
    Input("filter-organism", "value"),
    Input("filter-assay", "value"),
    Input("filter-mission", "value"),
    Input("filter-tissue", "value"),
    Input("year-slider", "value"),
    State("filters-state", "data"),
    prevent_initial_call=True
)
def update_filters(apply_clicks, clear_clicks, org, assay, mission, tissue, years, prev):
    trig = ctx.triggered_id
    if trig == "clear-filters":
        return {"organism": None, "assay": None, "mission": None, "tissue": None, "year_range": [2000, 2025]}
    fs = {
        "organism": org or None,
        "assay": assay or None,
        "mission": mission or None,
        "tissue": tissue or None,
        "year_range": years or [2000, 2025],
    }
    return fs

@app.callback(
    Output("search-results", "data"),
    Output("analytics-data", "data"),
    Output("results-count-badge", "children"),
    Input("search-btn", "n_clicks"),
    Input("query-input", "value"),
    Input("filters-state", "data"),
    State("backend-status", "data"),
    prevent_initial_call=True
)
def run_search(n_clicks, query, filters_state, backend_ok):
    if not backend_ok:
        return [], {}, "0"
    query = (query or "").strip()
    if not query:
        return [], {}, "0"
    body = {
        "query": query,
        "top_k": 25,
        "filters": filters_state or {"year_range": [2000, 2025]},
        "re_rank": True
    }
    try:
        res = api_post("/search", json_body=body, timeout=30)
        results = res if isinstance(res, list) else res.get("results", [])
    except Exception as e:
        results = []
    analytics = aggregate_for_analytics(results)
    return results, analytics, str(len(results))

@app.callback(
    Output("results-list", "children"),
    Input("search-results", "data")
)
def render_results_list(results):
    if not results:
        return [dbc.Alert("No results yet. Try a query above.", color="secondary")]
    cards = [render_result_card(item) for item in results]
    return cards

# Pattern-matching selection
@app.callback(
    Output("selected-ids", "data"),
    Input({"type":"result-select","id": ALL}, "n_clicks"),
    Input({"type":"result-check","id": ALL}, "value"),
    State("selected-ids", "data"),
    State("search-results", "data"),
    prevent_initial_call=True
)
def update_selection(title_clicks, check_values, selected_ids, results):
    selected_ids = set(selected_ids or [])
    # Title clicks
    if title_clicks:
        # find which was clicked
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "result-select":
            sid = str(triggered.get("id"))
            if sid in selected_ids:
                selected_ids.remove(sid)
            else:
                selected_ids.add(sid)
    # Checklist selections
    if check_values:
        # check_values is a list aligned with ALL pattern, merge all
        merged = set()
        for v in check_values:
            if isinstance(v, list):
                merged.update([str(x) for x in v])
        # Keep union with existing clicked items
        selected_ids = selected_ids.union(merged)
    return sorted(list(selected_ids))

@app.callback(
    Output("graph-data", "data"),
    Output("knowledge-graph", "elements"),
    Input("selected-ids", "data"),
    State("backend-status", "data"),
    prevent_initial_call=True
)
def build_graph_for_selection(selected_ids, backend_ok):
    if not backend_ok or not selected_ids:
        return {"nodes": [], "edges": []}, []
    try:
        body = {"ids": selected_ids, "scope": "local"}
        g = api_post("/graph", json_body=body, timeout=30)
        elements = to_cyto_elements(g)
        return g, elements
    except Exception:
        return {"nodes": [], "edges": []}, []

@app.callback(
    Output("knowledge-graph", "layout"),
    Input("graph-layout", "value")
)
def update_graph_layout(layout_name):
    return {"name": layout_name or "cose"}

@app.callback(
    Output("graph-node-details", "children"),
    Input("knowledge-graph", "selectedNodeData"),
    prevent_initial_call=True
)
def show_node_details(selected):
    if not selected:
        return dbc.Alert("Select a node to see details.", color="secondary")
    n = selected[0]
    label = n.get("label")
    typ = n.get("type")
    meta = n.get("meta", {})
    items = []
    for k, v in (meta.items() if isinstance(meta, dict) else []):
        items.append(html.Li([html.Strong(f"{k}: "), html.Span(str(v))]))
    return dbc.Card(dbc.CardBody([
        html.H6(f"{typ}: {label}"),
        html.Ul(items) if items else html.P("No additional metadata.")
    ]), className="mt-2")

@app.callback(
    Output("summary-output", "children"),
    Input("summarize-btn", "n_clicks"),
    State("selected-ids", "data"),
    State("question-input", "value"),
    State("summary-style", "value"),
    State("backend-status", "data"),
    prevent_initial_call=True
)
def run_summary(n_clicks, selected_ids, question, style, backend_ok):
    if not backend_ok:
        return "Backend not available."
    if not selected_ids:
        return "Select at least one item from search results."
    body = {
        "ids": selected_ids,
        "question": (question or "Summarize the key findings."),
        "style": (style or "abstract"),
    }
    try:
        res = api_post("/summarize", json_body=body, timeout=60)
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"Summarization failed: {e}"

@app.callback(
    Output("timeline-graph", "figure"),
    Output("organism-bar", "figure"),
    Output("assay-bar", "figure"),
    Output("mission-bar", "figure"),
    Output("tissue-top", "figure"),
    Input("analytics-data", "data")
)
def update_analytics(analytics):
    analytics = analytics or {}
    timeline_fig = timeline_fig_from_counts(analytics.get("timeline", {}))
    org_fig = analytics_fig_from_counts(analytics.get("organism", {}), "By organism")
    assay_fig = analytics_fig_from_counts(analytics.get("assay", {}), "By assay")
    mission_fig = analytics_fig_from_counts(analytics.get("mission", {}), "By mission")
    tissue_fig = analytics_fig_from_counts(analytics.get("tissue", {}), "Top tissues")
    return timeline_fig, org_fig, assay_fig, mission_fig, tissue_fig

# URL state: preload and persist
@app.callback(
    Output("query-input", "value"),
    Output("filter-organism", "value"),
    Output("filter-assay", "value"),
    Output("filter-mission", "value"),
    Output("filter-tissue", "value"),
    Output("year-slider", "value"),
    Input("url", "search"),
    prevent_initial_call=False
)
def preload_from_url(search):
    state = parse_url_query(search or "")
    q = state.get("q")
    filters = state.get("filters") or {}
    return (
        q,
        filters.get("organism"),
        filters.get("assay"),
        filters.get("mission"),
        filters.get("tissue"),
        filters.get("year_range") or [2000, 2025],
    )

@app.callback(
    Output("url", "search"),
    Input("query-input", "value"),
    Input("filters-state", "data"),
)
def persist_to_url(q, filters):
    state = {"q": (q or "").strip(), "filters": filters or {"year_range": [2000, 2025]}}
    return build_url_query(state)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

