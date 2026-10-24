"""Shared fixtures. Everything runs offline: hashing embeddings, temp dirs, mocked HTTP."""

from __future__ import annotations

from dataclasses import replace

import pytest

from telecom_agent.config import PROJECT_ROOT, get_settings
from telecom_agent.database import init_db
from telecom_agent.embeddings import HashingEmbeddings
from telecom_agent.ingest import ingest
from telecom_agent.retriever import KnowledgeBase

CHURN_URL = "http://churn.test"


@pytest.fixture(scope="session")
def embeddings():
    return HashingEmbeddings()


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("agent")
    return replace(
        get_settings(),
        llm_provider="fake",
        embedding_backend="hashing",
        router_mode="rules",
        kb_dir=PROJECT_ROOT / "data" / "knowledge_base",
        chroma_dir=tmp / "chroma",
        db_path=tmp / "customers.db",
        log_path=tmp / "runs.jsonl",
        churn_api_url=CHURN_URL,
        churn_api_timeout=1.0,
    )


@pytest.fixture(scope="session")
def indexed(settings, embeddings):
    """Ingest the real knowledge base once per test session."""
    summary = ingest(settings, embeddings)
    init_db(settings.db_path)
    return summary


@pytest.fixture
def kb(settings, embeddings, indexed):
    return KnowledgeBase(settings, embeddings)
