"""SQLAlchemy engine and session factory."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base


def _get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///aesthetic-ai.db")


def build_engine(database_url: str | None = None) -> Engine:
    """Create and configure the SQLAlchemy engine."""
    url = database_url or _get_database_url()
    connect_args = {}

    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        engine = create_engine(url, connect_args=connect_args)
        # Enable WAL mode for concurrent crawler + worker access
        @event.listens_for(engine, "connect")
        def set_wal_mode(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    else:
        engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

    return engine


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables if they do not exist. Returns the engine."""
    eng = engine or build_engine()
    Base.metadata.create_all(eng)
    return eng


# Module-level defaults — replaced in tests via dependency injection
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = build_engine()
        init_db(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session and handles commit/rollback."""
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def override_engine(engine: Engine) -> None:
    """Replace the module-level engine and session factory — used in tests."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
