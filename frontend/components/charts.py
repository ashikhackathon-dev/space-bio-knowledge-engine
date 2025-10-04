# frontend/components/charts.py
import pandas as pd
import plotly.express as px

def _blank(title):
    fig = px.bar(title=title)
    fig.update_layout(template="plotly_dark", annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
    return fig

def organism_bar(df: pd.DataFrame):
    if df is None or df.empty or "organism" not in df.columns:
        return _blank("By organism")
    s = df["organism"].dropna().astype(str).value_counts()
    fig = px.bar(x=s.index, y=s.values, title="By organism")
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

def assay_bar(df: pd.DataFrame):
    if df is None or df.empty or "assay" not in df.columns:
        return _blank("By assay")
    s = df["assay"].dropna().astype(str).value_counts()
    fig = px.bar(x=s.index, y=s.values, title="By assay")
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

def mission_bar(df: pd.DataFrame):
    if df is None or df.empty or "mission" not in df.columns:
        return _blank("By mission")
    s = df["mission"].dropna().astype(str).value_counts()
    fig = px.bar(x=s.index, y=s.values, title="By mission")
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

def top_tissues(df: pd.DataFrame, k=10):
    if df is None or df.empty or "tissue" not in df.columns:
        return _blank("Top tissues")
    s = df["tissue"].dropna().astype(str).value_counts().head(k)
    fig = px.bar(x=s.values, y=s.index, orientation="h", title="Top tissues")
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

