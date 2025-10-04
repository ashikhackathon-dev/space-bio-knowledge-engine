# frontend/components/timeline.py
import pandas as pd
import plotly.express as px

def timeline_plot(df: pd.DataFrame, title="Studies over time"):
    if df is None or df.empty or "year" not in df.columns:
        fig = px.bar(title=title)
        fig.update_layout(template="plotly_dark", annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        return fig
    try:
        s = df["year"].dropna().astype(int).value_counts().sort_index()
    except Exception:
        fig = px.bar(title=title)
        fig.update_layout(template="plotly_dark", annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        return fig
    fig = px.bar(x=s.index.astype(str), y=s.values, title=title)
    fig.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title=None)
    return fig

