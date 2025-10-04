# %% [markdown]
# # Space Biology Data Cleaning

# %%
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio

load_dotenv()
CWD = Path(os.getcwd()).resolve()
REPO_ROOT = CWD if (CWD.name != "notebooks") else CWD.parent
DATA_DIR = REPO_ROOT / "data"
RAW_CSV = DATA_DIR / "experiments.csv"
CLEAN_CSV = DATA_DIR / "experiments.csv"
pio.templates.default = "plotly_dark"
BACKEND_BASE = os.environ.get("FRONTEND_BACKEND_URL", "http://localhost:5000")
OSDR_BASE = os.environ.get("OSDR_BASE", "")
print("Repo root:", REPO_ROOT, "| Data dir:", DATA_DIR)

# %% [markdown]
# ## Load data

# %%
if not RAW_CSV.exists():
    raise FileNotFoundError(f"Missing {RAW_CSV}. Place experiments.csv in data/.")
df = pd.read_csv(RAW_CSV)
print("Rows loaded:", len(df))
df.head(3)

# %% [markdown]
# ## Cleaning utilities

# %%
def _clean_text(s: Any) -> str:
    if pd.isna(s): return ""
    t = str(s).replace("\\u00a0", " ")
    t = re.sub(r"[\\r\\t]+", " ", t)
    t = re.sub(r"\\s+", " ", t)
    return t.strip()

ORGANISM_MAP = {"mus musculus":"Mus musculus","mouse":"Mus musculus","mice":"Mus musculus",
                "arabidopsis":"Arabidopsis thaliana","arabidopsis thaliana":"Arabidopsis thaliana",
                "drosophila":"Drosophila melanogaster"}
ASSAY_MAP = {"rna-seq":"RNA-seq","rnaseq":"RNA-seq","microarray":"microarray","amplicon":"amplicon","methyl-seq":"methyl-seq"}
MISSION_MAP = {"iss":"ISS","international space station":"ISS"}
TISSUE_MAP = {"bone":"bone","femur":"femur","leaf":"leaf"}

def std_label(s: str, m: Dict[str,str]) -> str | None:
    if not s: return None
    low = s.strip().lower()
    return m.get(low, s.strip())

def parse_metadata_json(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict): return x
    if pd.isna(x) or x is None or str(x).strip() == "": return {}
    try: return json.loads(x)
    except Exception: return {}

def coerce_year(v: Any) -> int | None:
    try:
        iv = int(v)
        if 1900 <= iv <= 2100: return iv
    except Exception:
        pass
    return None

# %% [markdown]
# ## Apply cleaning and standardization

# %%
for c in ["study_accession","publication_title","year","mission","organism","tissue",
          "assay","platform","environment","doi","source_url","abstract","metadata_json"]:
    if c not in df.columns: df[c] = None

for c in ["publication_title","mission","organism","tissue","assay","platform","environment","doi","source_url","abstract"]:
    df[c] = df[c].map(_clean_text)

df["year"] = df["year"].map(coerce_year)
df["organism_std"] = df["organism"].map(lambda x: std_label(x, ORGANISM_MAP))
df["assay_std"]   = df["assay"].map(lambda x: std_label(x, ASSAY_MAP))
df["mission_std"] = df["mission"].map(lambda x: std_label(x, MISSION_MAP))
df["tissue_std"]  = df["tissue"].map(lambda x: std_label(x, TISSUE_MAP))
df["metadata_parsed"] = df["metadata_json"].map(parse_metadata_json)
df.head(5)

# %% [markdown]
# ## Deduplicate

# %%
before = len(df)
df = df.drop_duplicates(subset=["study_accession","assay","tissue"]).reset_index(drop=True)
after = len(df)
print(f"Deduplicated: {before} -> {after}")
df.head(3)

# %% [markdown]
# ## Derive filter columns and checks

# %%
def infer_year_from_doi(doi: str) -> int | None:
    if not doi: return None
    m = re.search(r"(20\\d{2}|19\\d{2})", doi)
    return int(m.group(1)) if m else None

mask_missing_year = df["year"].isna()
df.loc[mask_missing_year, "year"] = df.loc[mask_missing_year, "doi"].map(infer_year_from_doi)
key_cols = ["study_accession","publication_title","year","mission_std","organism_std","tissue_std","assay_std"]
df[key_cols].isna().sum()

# %% [markdown]
# ## Save cleaned CSV

# %%
CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_CSV, index=False)
print("Saved:", CLEAN_CSV, "rows:", len(df))

# %% [markdown]
# ## Quick EDA

# %%
def bar_from_counts(series: pd.Series, title: str):
    if series is None or series.empty:
        fig = px.bar(title=title); fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5); return fig
    s = series.dropna().astype(str).value_counts()
    fig = px.bar(x=s.index, y=s.values, title=title); fig.update_layout(xaxis_title=None, yaxis_title=None); return fig

fig_org = bar_from_counts(df["organism_std"], "By organism"); fig_org.show()
fig_assay = bar_from_counts(df["assay_std"], "By assay"); fig_assay.show()
fig_mission = bar_from_counts(df["mission_std"], "By mission"); fig_mission.show()
fig_tissue = bar_from_counts(df["tissue_std"], "Top tissues"); fig_tissue.show()

if "year" in df.columns:
    y = df["year"].dropna()
    try:
        y = y.astype(int); yc = y.value_counts().sort_index()
        fig_y = px.bar(x=yc.index.astype(str), y=yc.values, title="Studies over time")
        fig_y.update_layout(xaxis_title="Year", yaxis_title="Count"); fig_y.show()
    except Exception: pass

# %% [markdown]
# ## Optional: OSDR fetch stub and backend ingest

# %%
import httpx

def fetch_from_osdr(query: str, limit: int = 25) -> pd.DataFrame:
    base = (OSDR_BASE or "").strip()
    if not base:
        print("OSDR_BASE not set; skipping remote fetch.")
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    return pd.DataFrame(rows)

def post_to_backend_ingest(csv_path: Path, mode: str = "append") -> dict:
    url = f"{BACKEND_BASE.rstrip('/')}/ingest"
    payload = {"sources": [str(csv_path)], "mode": mode}
    try:
        r = httpx.post(url, json=payload, timeout=60); r.raise_for_status(); return r.json()
    except Exception as e:
        print("Ingest failed:", e); return {"status": "error", "message": str(e)}

