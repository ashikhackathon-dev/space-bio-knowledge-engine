from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


# Naming convention for constraints (important for SQLite with Alembic compatibility)
metadata_obj = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


class Base(DeclarativeBase):
    metadata = metadata_obj


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    organism: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    assay: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    mission: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    tissue: Mapped[Optional[str]] = mapped_column(String(128), index=True)

    source_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[List[Chunk]] = relationship("Chunk", back_populates="publication", cascade="all, delete-orphan")  # type: ignore # noqa: F821


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    publication: Mapped[Optional[Publication]] = relationship(
        "Publication", back_populates="chunks", primaryjoin="foreign(Chunk.doc_id)==Publication.doc_id"
    )


Index("ix_chunks_doc_id_chunk", Chunk.doc_id, Chunk.chunk_id)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    chunk_id: Mapped[str] = mapped_column(String(255), index=True)
    ent_type: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(String(512))
    normalized: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    source_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    target_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    rel_type: Mapped[str] = mapped_column(String(64), index=True)


# --- Engine / Session ---


def get_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or os.getenv("DB_URL", "sqlite:///spacebio.db")
    connect_args = {}
    if url.startswith("sqlite"):  # allow multi-thread in FastAPI
        connect_args = {"check_same_thread": False}
    engine = create_engine(url, future=True, echo=False, connect_args=connect_args)
    return engine


def get_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    ensure_fts_tables(engine)


# --- SQLite FTS5 support for BM25 hybrid search ---


def ensure_fts_tables(engine: Engine) -> None:
    """Create FTS5 virtual table if not exists.

    We keep it simple and do not use content synchronization triggers. Ingestion
    will explicitly (re)populate this table.
    """
    with engine.begin() as conn:
        # Enable FTS5 extension implicitly available in SQLite builds
        conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(doc_id, chunk_id, text, tokenize='porter');"
        )


def reset_fts(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM chunk_fts;")


def populate_fts(engine: Engine, rows: Iterable[Tuple[str, str, str]]) -> int:
    """Bulk insert rows into FTS table.

    Args:
        rows: iterable of (doc_id, chunk_id, text)
    Returns:
        count inserted
    """
    count = 0
    with engine.begin() as conn:
        for doc_id, chunk_id, text_value in rows:
            conn.exec_driver_sql(
                "INSERT INTO chunk_fts(doc_id, chunk_id, text) VALUES (?, ?, ?)",
                (doc_id, chunk_id, text_value),
            )
            count += 1
    return count


def fts_search(engine: Engine, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Run BM25-ranked FTS search over chunks."""
    sql = (
        "SELECT doc_id, chunk_id, snippet(chunk_fts, 2, '[', ']', '…', 16) as snippet, "
        "bm25(chunk_fts) as bm25_score FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25_score LIMIT ?"
    )
    with engine.begin() as conn:
        res = conn.exec_driver_sql(sql, (query, limit))
        rows = [dict(r._mapping) for r in res]
    return rows


# Convenience dataclass for returning brief stats
@dataclass
class DbStats:
    publication_count: int
    chunk_count: int
    entity_count: int
    relation_count: int
    fts_count: int


def get_db_stats(engine: Engine) -> DbStats:
    with engine.begin() as conn:
        publication_count = conn.exec_driver_sql("SELECT COUNT(*) FROM publications").scalar_one()
        chunk_count = conn.exec_driver_sql("SELECT COUNT(*) FROM chunks").scalar_one()
        entity_count = conn.exec_driver_sql("SELECT COUNT(*) FROM entities").scalar_one()
        relation_count = conn.exec_driver_sql("SELECT COUNT(*) FROM relations").scalar_one()
        try:
            fts_count = conn.exec_driver_sql("SELECT COUNT(*) FROM chunk_fts").scalar_one()
        except Exception:
            fts_count = 0
    return DbStats(
        publication_count=publication_count,
        chunk_count=chunk_count,
        entity_count=entity_count,
        relation_count=relation_count,
        fts_count=fts_count,
    )



