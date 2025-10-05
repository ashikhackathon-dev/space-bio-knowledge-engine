from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

import utils
from ai import entity_extractor as ie
from ai import search as search_ai
from ai import summarizer
from database import (
    Chunk,
    Entity,
    Publication,
    Relation,
    create_all,
    fts_search,
    get_db_stats,
    get_engine,
    get_sessionmaker,
    populate_fts,
    reset_fts,
)



class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    DB_URL: str = "sqlite:///spacebio.db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIR: str = "./vectors"
    OSDR_BASE: Optional[str] = None
    NSLSL_BASE: Optional[str] = None
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()  # reads from .env if present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spacebio.api")


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    index_dim: int
    index_size: int
    db: Dict[str, int]


class IngestRequest(BaseModel):
    sources: List[str]
    mode: str = Field("append", pattern="^(append|rebuild)$")


class SearchFilters(BaseModel):
    organism: Optional[str] = None
    assay: Optional[str] = None
    mission: Optional[str] = None
    year_range: Optional[Tuple[int, int]] = None
    tissue: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[SearchFilters] = None
    re_rank: bool = True
    re_rank_model: Optional[str] = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")


class SearchHit(BaseModel):
    doc_id: str
    chunk_id: str
    title: Optional[str]
    snippet: Optional[str]
    score: float
    filters_matched: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class SummarizeRequest(BaseModel):
    ids: List[str]
    question: Optional[str] = None
    style: str = Field("bullet", pattern="^(bullet|abstract|methods|clinician)$")


class SummaryResponse(BaseModel):
    answer: str
    citations: List[str]


class EntitiesRequest(BaseModel):
    ids: Optional[List[str]] = None
    query: Optional[str] = None


class EntitySpan(BaseModel):
    ent_type: str
    text: str
    normalized: Optional[str]
    start: Optional[int]
    end: Optional[int]


class EntitiesResponse(BaseModel):
    entities: List[EntitySpan]


class GraphRequest(BaseModel):
    ids: Optional[List[str]] = None
    query: Optional[str] = None
    scope: str = Field("local", pattern="^(local|global)$")


class GraphNode(BaseModel):
    id: str
    label: str
    type: str


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class PublicationsResponse(BaseModel):
    total: int
    items: List[Dict[str, Any]]


def create_app() -> FastAPI:
    app = FastAPI(title="Space Biology KB API", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- State: DB and FAISS ---
    engine = get_engine(settings.DB_URL)
    SessionLocal = get_sessionmaker(engine)
    create_all(engine)

    # Embeddings
    embedding_model_name = settings.EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(embedding_model_name)
        embedding_dim = int(embedder.get_sentence_embedding_dimension())
    except Exception as e:
        logger.warning("Failed to initialize embedding model %s: %s", embedding_model_name, e)
        embedder = None
        embedding_dim = 384  # typical for MiniLM

    os.makedirs(settings.VECTOR_DIR, exist_ok=True)
    vx = search_ai.load_faiss_index(settings.VECTOR_DIR, embedding_dim) or search_ai.create_empty_faiss(embedding_dim)

    # ID mapping persistence
    idmap_path = os.path.join(settings.VECTOR_DIR, "idmap.json")
    try:
        if os.path.exists(idmap_path):
            with open(idmap_path, "r", encoding="utf-8") as f:
                faiss_ids: List[str] = json.load(f)
        else:
            faiss_ids = []
    except Exception:
        faiss_ids = []

    # --- Error handlers ---
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # --- Routes ---
    @app.get("/health", response_model=HealthResponse)
    async def health():
        stats = get_db_stats(engine)
        index_size = vx.index.ntotal if vx and vx.index is not None else 0
        return HealthResponse(
            status="ok",
            embedding_model=embedding_model_name,
            index_dim=vx.dim if vx else embedding_dim,
            index_size=index_size,
            db={
                "publications": stats.publication_count,
                "chunks": stats.chunk_count,
                "entities": stats.entity_count,
                "relations": stats.relation_count,
                "fts_rows": stats.fts_count,
            },
        )

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        if req.mode == "rebuild":
            # Reset DB tables minimally and FTS. Keep schema.
            with SessionLocal() as s:
                s.query(Relation).delete()
                s.query(Entity).delete()
                s.query(Chunk).delete()
                s.query(Publication).delete()
                s.commit()
            reset_fts(engine)
            if vx.index.ntotal > 0:
                # Recreate empty index
                vx.index.reset()
                faiss_ids.clear()
                try:
                    with open(idmap_path, "w", encoding="utf-8") as f:
                        json.dump(faiss_ids, f)
                except Exception:
                    pass

        if embedder is None:
            raise HTTPException(status_code=422, detail="Embedding model not initialized")

        all_rows_for_fts: List[Tuple[str, str, str]] = []
        added_count = 0
        with SessionLocal() as s:
            for source in req.sources:
                try:
                    doc_id, text = utils.resolve_source_to_text(source)
                except Exception as e:
                    logger.warning("Failed to read source %s: %s", source, e)
                    continue

                chunks = utils.chunk_text(text)
                publication = s.query(Publication).filter_by(doc_id=doc_id).first()
                if publication is None:
                    publication = Publication(doc_id=doc_id, title=doc_id)
                    s.add(publication)
                    s.flush()

                # Upsert chunks
                vectors: List[np.ndarray] = []
                ids: List[str] = []
                for idx, chunk in enumerate(chunks):
                    chunk_id = utils.compute_chunk_id(doc_id, idx)
                    existing = s.query(Chunk).filter_by(chunk_id=chunk_id).first()
                    if existing is None:
                        row = Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk, metadata_json={})
                        s.add(row)
                        added_count += 1
                    ids.append(chunk_id)
                    all_rows_for_fts.append((doc_id, chunk_id, chunk))
                    vectors.append(embedder.encode([chunk], convert_to_numpy=True, normalize_embeddings=True)[0])

                if vectors:
                    vec_array = np.vstack(vectors)
                    search_ai.add_to_index(vec_array, ids, vx)
                    faiss_ids.extend(ids)
                    try:
                        search_ai.persist_faiss_index(settings.VECTOR_DIR, vx)
                        with open(idmap_path, "w", encoding="utf-8") as f:
                            json.dump(faiss_ids, f)
                    except Exception as e:
                        logger.warning("Failed to persist FAISS or idmap: %s", e)

                # Entities and relations (simple heuristic)
                all_entities = ie.simple_rule_entities(text)
                persisted_entities: List[Entity] = []
                for ent in all_entities:
                    e_row = Entity(
                        doc_id=doc_id,
                        chunk_id=ids[0] if ids else doc_id,
                        ent_type=ent.ent_type,
                        text=ent.text,
                        normalized=ent.normalized,
                        start=ent.start,
                        end=ent.end,
                    )
                    s.add(e_row)
                    s.flush()
                    persisted_entities.append(e_row)
                # naive linking by first matching text
                def find_entity_id(text: str) -> Optional[int]:
                    for e in persisted_entities:
                        if e.text == text:
                            return e.id
                    return None
                for rel in ie.simple_relations(all_entities):
                    src_id = find_entity_id(rel.source_text)
                    tgt_id = find_entity_id(rel.target_text)
                    if src_id and tgt_id:
                        s.add(
                            Relation(
                                doc_id=doc_id,
                                source_entity_id=src_id,
                                target_entity_id=tgt_id,
                                rel_type=rel.rel_type,
                            )
                        )

                s.commit()

        if all_rows_for_fts:
            # Only add to FTS rows that are new (those we inserted as chunks)
            populate_fts(engine, all_rows_for_fts)

        return {"status": "ok", "added_chunks": added_count}

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        if embedder is None:
            raise HTTPException(status_code=422, detail="Embedding model not initialized")

        # Vector search
        vec = embedder.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = vx.search(vec, top_k=req.top_k * 5)
        # Build ranking list of (doc_id, score)
        vector_rank: List[Tuple[str, float]] = []
        with SessionLocal() as s:
            for idx in I[0].tolist():
                if idx < 0 or idx >= len(faiss_ids):
                    continue
                chunk_id = faiss_ids[idx]
                ch = s.query(Chunk).filter_by(chunk_id=chunk_id).first()
                if ch is None:
                    continue
                vector_rank.append((ch.doc_id, float(1.0)))

        # FTS BM25
        bm25_rows = fts_search(engine, req.query, limit=req.top_k * 5)
        bm25_rank = [(r["doc_id"], float(1.0)) for r in bm25_rows]

        fused = search_ai.reciprocal_rank_fusion([vector_rank, bm25_rank])
        top_doc_ids = [doc_id for doc_id, _ in fused[: req.top_k]]

        hits: List[SearchHit] = []
        with SessionLocal() as s:
            for doc_id in top_doc_ids:
                pub = s.query(Publication).filter_by(doc_id=doc_id).first()
                chunk = s.query(Chunk).filter_by(doc_id=doc_id).first()
                snippet = None
                if chunk is not None:
                    snippet = (chunk.text or "")[:320]
                filters_matched: Dict[str, Any] = {}
                if req.filters:
                    filter_failed = False
                    for f in ["organism", "assay", "mission", "tissue"]:
                        val = getattr(req.filters, f)
                        if val:
                            if pub and getattr(pub, f) == val:
                                filters_matched[f] = True
                            else:
                                filter_failed = True
                                break
                    if not filter_failed and req.filters.year_range and pub and pub.year:
                        y0, y1 = req.filters.year_range
                        if not (y0 <= pub.year <= y1):
                            filter_failed = True
                    if filter_failed:
                        continue
                hits.append(
                    SearchHit(
                        doc_id=doc_id,
                        chunk_id=chunk.chunk_id if chunk else doc_id,
                        title=pub.title if pub else doc_id,
                        snippet=snippet,
                        score=1.0,
                        filters_matched=filters_matched,
                    )
                )

        # Optional re-ranking using a cross-encoder (if available)
        if req.re_rank:
            try:
                from sentence_transformers import CrossEncoder

                model_name = req.re_rank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ce = CrossEncoder(model_name)
                pairs = [(req.query, h.snippet or h.title or h.doc_id) for h in hits]
                scores = ce.predict(pairs)
                for h, sc in zip(hits, scores):
                    h.score = float(sc)
                hits.sort(key=lambda h: h.score, reverse=True)
            except Exception as e:
                logger.warning("Cross-encoder rerank failed: %s", e)

        return SearchResponse(hits=hits)

    @app.post("/summarize", response_model=SummaryResponse)
    async def summarize(req: SummarizeRequest):
        with SessionLocal() as s:
            texts_with_ids: List[Tuple[str, str]] = []
            for doc_id in req.ids:
                chunk = s.query(Chunk).filter_by(doc_id=doc_id).first()
                if chunk is None:
                    continue
                texts_with_ids.append((doc_id, chunk.text))

        if not texts_with_ids:
            raise HTTPException(status_code=422, detail="No documents found for provided ids")

        prompt = summarizer.format_prompt(req.style, req.question or "Summarize", texts_with_ids)

        # Try OpenAI, fallback to simple concatenation
        answer = None
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": summarizer.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=700,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                logger.warning("OpenAI call failed: %s", e)

        if answer is None:
            # Minimal heuristic summary
            bullets = [f"- {text[:200]} [DOC:{doc_id}]" for doc_id, text in texts_with_ids]
            answer = "\n".join(bullets)

        citations = [doc_id for doc_id, _ in texts_with_ids]
        return SummaryResponse(answer=answer, citations=citations)

    @app.post("/entities", response_model=EntitiesResponse)
    async def entities(req: EntitiesRequest):
        results: List[EntitySpan] = []
        with SessionLocal() as s:
            texts: List[Tuple[str, str]] = []
            if req.ids:
                for doc_id in req.ids:
                    chunk = s.query(Chunk).filter_by(doc_id=doc_id).first()
                    if chunk is not None:
                        texts.append((doc_id, chunk.text))
            elif req.query:
                texts.append(("query", req.query))

        for doc_id, text in texts:
            ents = ie.simple_rule_entities(text)
            for ent in ents:
                results.append(
                    EntitySpan(
                        ent_type=ent.ent_type,
                        text=ent.text,
                        normalized=ent.normalized,
                        start=ent.start,
                        end=ent.end,
                    )
                )

        return EntitiesResponse(entities=results)

    @app.post("/graph", response_model=GraphResponse)
    async def graph(req: GraphRequest):
        nodes: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        def ensure_node(node_id: str, label: str, type_: str) -> None:
            if node_id not in nodes:
                nodes[node_id] = GraphNode(id=node_id, label=label, type=type_)

        with SessionLocal() as s:
            doc_ids: List[str] = []
            if req.ids:
                doc_ids = req.ids
            elif req.query:
                bm25_rows = fts_search(engine, req.query, limit=20)
                doc_ids = list({r["doc_id"] for r in bm25_rows})

            for doc_id in doc_ids:
                pub = s.query(Publication).filter_by(doc_id=doc_id).first()
                ensure_node(doc_id, pub.title if pub and pub.title else doc_id, "Publication")
                es = s.query(Entity).filter_by(doc_id=doc_id).all()
                for e in es:
                    ensure_node(f"{doc_id}:{e.text}", e.text, e.ent_type)
                    edges.append(GraphEdge(source=doc_id, target=f"{doc_id}:{e.text}", type="MENTIONS"))

        return GraphResponse(nodes=list(nodes.values()), edges=edges)

    @app.get("/publications", response_model=PublicationsResponse)
    async def publications(page: int = 1, page_size: int = 20):
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        with SessionLocal() as s:
            total = s.query(Publication).count()
            items = (
                s.query(Publication)
                .order_by(Publication.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            payload = [
                {
                    "doc_id": p.doc_id,
                    "title": p.title,
                    "year": p.year,
                    "organism": p.organism,
                    "assay": p.assay,
                    "mission": p.mission,
                    "tissue": p.tissue,
                    "source_path": p.source_path,
                    "source_url": p.source_url,
                }
                for p in items
            ]
        return PublicationsResponse(total=total, items=payload)

    return app


app = create_app()



