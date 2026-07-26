"""
database/sqlite.py

SQLite database layer for Secure RAG.

Python 3.12+
SQLAlchemy 2.x
"""

from __future__ import annotations

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)

from config import settings


# =============================================================================
# Base
# =============================================================================

class Base(DeclarativeBase):
    """Base class for ORM models."""
    pass


# =============================================================================
# Models
# =============================================================================

class User(Base):
    """
    User accounts.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(30),
        default="guest",
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


class Document(Base):
    """
    Indexed documents.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
    )

    owner: Mapped[str] = mapped_column(
        String(50),
    )

    classification: Mapped[str] = mapped_column(
        String(20),
        default="internal",
    )

    indexed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


class AuditLog(Base):
    """
    Audit events.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    username: Mapped[str] = mapped_column(
        String(50),
    )

    action: Mapped[str] = mapped_column(
        String(100),
    )

    details: Mapped[str] = mapped_column(
        Text,
    )

    success: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


# =============================================================================
# Engine
# =============================================================================

engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# =============================================================================
# Helpers
# =============================================================================

def create_database() -> None:
    """
    Create database schema.
    """
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency.
    """
    session = SessionLocal()

    try:
        yield session

    finally:
        session.close()


# =============================================================================
# CRUD helpers
# =============================================================================

def create_user(
    session: Session,
    username: str,
    password_hash: str,
    role: str = "guest",
) -> User:

    user = User(
        username=username,
        password_hash=password_hash,
        role=role,
    )

    session.add(user)
    session.commit()
    session.refresh(user)

    return user


def get_user(
    session: Session,
    username: str,
) -> User | None:

    return (
        session.query(User)
        .filter(User.username == username)
        .first()
    )


def add_document(
    session: Session,
    filename: str,
    sha256: str,
    file_type: str,
    owner: str,
) -> Document:

    document = Document(
        filename=filename,
        sha256=sha256,
        file_type=file_type,
        owner=owner,
    )

    session.add(document)
    session.commit()
    session.refresh(document)

    return document


def write_audit(
    session: Session,
    username: str,
    action: str,
    details: str,
    success: bool = True,
) -> AuditLog:

    log = AuditLog(
        username=username,
        action=action,
        details=details,
        success=success,
    )

    session.add(log)
    session.commit()
    session.refresh(log)

    return log