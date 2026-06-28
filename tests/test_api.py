"""
tests/test_api.py
-----------------
Integration tests for the FastAPI endpoints.
ChromaDB and Ollama calls are mocked — no live services required.

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch ChromaDB before importing app so lifespan doesn't crash
with patch("chromadb.PersistentClient"):
    from api import app, _state

from models import OllamaStatus

client = TestClient(app)

MOCK_HIT = {
    "id": "pb-001",
    "document": "Title: SSH Brute Force\nSeverity: HIGH",
    "metadata": {"title": "SSH Brute Force Attack", "severity": "HIGH"},
    "distance": 0.05,
}

VALID_ANALYSIS = {
    "attack_type": "SSH Brute Force",
    "severity": "HIGH",
    "explanation": "Repeated SSH auth failures from single source.",
    "mitigation": ["Block IP at firewall", "Enable MFA"],
    "detection_recommendation": ">5 failures/60s from same IP",
    "mitre_attack": "T1110.001 - Brute Force: Password Guessing",
}


@pytest.fixture(autouse=True)
def mock_chroma_collection():
    """Inject a mock ChromaDB collection into app state for every test."""
    mock_col = MagicMock()
    mock_col.count.return_value = 20
    mock_col.query.return_value = {
        "ids":       [["pb-001"]],
        "documents": [["Title: SSH Brute Force\nSeverity: HIGH"]],
        "metadatas": [[{"title": "SSH Brute Force Attack", "severity": "HIGH"}]],
        "distances": [[0.05]],
    }
    _state.collection = mock_col
    yield mock_col
    _state.collection = None


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_ok_when_all_up():
    with patch("api._ollama_status") as mock_oll:
        mock_oll.return_value = OllamaStatus(
            reachable=True,
            available_models=["llama3", "nomic-embed-text"],
            required_models_present=True,
        )
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chroma_store_ready"] is True
    assert body["playbook_count"] == 20


def test_health_degraded_when_ollama_down():
    with patch("api._ollama_status") as mock_oll:
        mock_oll.return_value = OllamaStatus(
            reachable=False,
            available_models=[],
            required_models_present=False,
        )
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("degraded", "error")


# ── POST /analyze ─────────────────────────────────────────────────────────────

def test_analyze_returns_structured_response():
    with patch("rag.embed", return_value=[0.0] * 768):
        with patch("rag.generate", return_value=json.dumps(VALID_ANALYSIS)):
            response = client.post("/analyze", json={"alert": "Multiple SSH failures from 45.33.32.156"})

    assert response.status_code == 200
    body = response.json()
    assert "analysis" in body
    assert body["analysis"]["severity"] == "HIGH"
    assert body["analysis"]["mitre_attack"].startswith("T1110")
    assert len(body["analysis"]["mitigation"]) > 0
    assert len(body["retrieved_playbooks"]) > 0


def test_analyze_returns_422_on_unparseable_json():
    with patch("rag.embed", return_value=[0.0] * 768):
        with patch("rag.generate", return_value="definitely not json"):
            response = client.post("/analyze", json={"alert": "test"})
    assert response.status_code == 422


def test_analyze_respects_model_field():
    captured = {}

    def mock_generate(prompt, model="llama3", system=""):
        captured["model"] = model
        return json.dumps(VALID_ANALYSIS)

    with patch("rag.embed", return_value=[0.0] * 768):
        with patch("rag.generate", side_effect=mock_generate):
            client.post("/analyze", json={"alert": "test", "model": "mistral"})

    assert captured.get("model") == "mistral"


def test_analyze_top_k_bounds():
    """top_k outside [1, 6] must be rejected at the schema level."""
    response = client.post("/analyze", json={"alert": "test", "top_k": 0})
    assert response.status_code == 422

    response = client.post("/analyze", json={"alert": "test", "top_k": 7})
    assert response.status_code == 422


def test_analyze_empty_alert_rejected():
    response = client.post("/analyze", json={"alert": ""})
    assert response.status_code == 422


# ── GET /playbooks ────────────────────────────────────────────────────────────

def test_playbooks_returns_list():
    with patch("api._load_playbooks", return_value=[
        {
            "id": "pb-001", "title": "SSH Brute Force Attack", "severity": "HIGH",
            "mitre_technique": "T1110.001", "mitre_tactic": "Credential Access",
            "description": "Repeated SSH failures.", "indicators": ["Many failed logins"],
            "detection_rule": ">5 failures in 60s", "response_steps": ["Block IP"],
        }
    ]):
        response = client.get("/playbooks")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["playbooks"][0]["id"] == "pb-001"


# ── POST /ingest ──────────────────────────────────────────────────────────────

def test_ingest_returns_early_when_store_populated():
    """Should not re-embed if store has documents and force=False."""
    response = client.post("/ingest", json={"force": False})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "already contains" in body["message"]


# ── CORS headers ──────────────────────────────────────────────────────────────

def test_cors_allows_localhost_5173():
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "*"


def test_cors_allows_localhost_other_port():
    response = client.options(
        "/playbooks",
        headers={"Origin": "http://localhost:5174", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "*"


def test_cors_allows_unknown_origin_in_dev():
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.headers.get("access-control-allow-origin") == "*"