"""
tests/test_rag.py
-----------------
Unit tests for the core RAG pipeline functions.
Does NOT require Ollama or ChromaDB — all external calls are mocked.

Run:
    pytest tests/test_rag.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag


# ── parse_response ────────────────────────────────────────────────────────────

VALID_JSON = {
    "attack_type": "SSH Brute Force",
    "severity": "HIGH",
    "explanation": "Repeated SSH failures indicate brute force.",
    "mitigation": ["Block IP", "Enable MFA"],
    "detection_recommendation": ">5 failures/60s from same IP",
    "mitre_attack": "T1110.001 - Brute Force: Password Guessing",
}


def test_parse_response_clean_json():
    raw = json.dumps(VALID_JSON)
    result = rag.parse_response(raw)
    assert result["attack_type"] == "SSH Brute Force"
    assert result["severity"] == "HIGH"


def test_parse_response_strips_markdown_fences():
    raw = "```json\n" + json.dumps(VALID_JSON) + "\n```"
    result = rag.parse_response(raw)
    assert result["severity"] == "HIGH"


def test_parse_response_strips_leading_text():
    raw = "Sure! Here is the JSON:\n" + json.dumps(VALID_JSON)
    result = rag.parse_response(raw)
    assert result["attack_type"] == "SSH Brute Force"


def test_parse_response_raises_on_invalid():
    with pytest.raises(json.JSONDecodeError):
        rag.parse_response("This is not JSON at all.")


# ── build_prompt ──────────────────────────────────────────────────────────────

def _make_hit(title: str, severity: str, document: str = "doc text") -> rag.Hit:
    return {
        "id": "pb-001",
        "document": document,
        "metadata": {"title": title, "severity": severity},
        "distance": 0.1,
    }


def test_build_prompt_contains_alert():
    hits = [_make_hit("SSH Brute Force Attack", "HIGH")]
    prompt = rag.build_prompt("Multiple SSH failures", hits)
    assert "Multiple SSH failures" in prompt


def test_build_prompt_contains_playbook_title():
    hits = [_make_hit("SSH Brute Force Attack", "HIGH")]
    prompt = rag.build_prompt("test", hits)
    assert "SSH Brute Force Attack" in prompt


def test_build_prompt_multiple_hits():
    hits = [
        _make_hit("SSH Brute Force Attack", "HIGH", "doc1"),
        _make_hit("Port Scanning", "MEDIUM", "doc2"),
    ]
    prompt = rag.build_prompt("test alert", hits)
    assert "doc1" in prompt
    assert "doc2" in prompt


# ── retrieve ──────────────────────────────────────────────────────────────────

def test_retrieve_returns_correct_hit_structure():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids":       [["pb-001", "pb-002"]],
        "documents": [["doc A", "doc B"]],
        "metadatas": [[{"title": "SSH BF", "severity": "HIGH"},
                       {"title": "Port Scan", "severity": "MEDIUM"}]],
        "distances": [[0.05, 0.32]],
    }

    with patch.object(rag, "embed", return_value=[0.1] * 768):
        hits = rag.retrieve("test query", mock_collection, top_k=2)

    assert len(hits) == 2
    assert hits[0]["id"] == "pb-001"
    assert hits[0]["distance"] == 0.05
    assert hits[1]["metadata"]["title"] == "Port Scan"


# ── generate (model parameter threading) ─────────────────────────────────────

def test_generate_passes_model_to_ollama():
    mock_response = {"message": {"content": json.dumps(VALID_JSON)}}

    with patch("ollama.chat", return_value=mock_response) as mock_chat:
        result = rag.generate("test prompt", model="mistral")

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs[1]["model"] == "mistral" or call_kwargs[0][0] == "mistral" \
        or call_kwargs.kwargs.get("model") == "mistral"


def test_generate_no_global_mutation():
    """Calling generate() with different models must not mutate module-level state."""
    mock_response = {"message": {"content": "{}"}}
    with patch("ollama.chat", return_value=mock_response):
        rag.generate("prompt1", model="llama3")
        rag.generate("prompt2", model="mistral")
    # There is no _llm_model global to mutate in the refactored rag.py —
    # this test confirms the module has no such attribute.
    assert not hasattr(rag, "_llm_model"), (
        "_llm_model global still exists — model must be passed as a parameter"
    )


# ── run_query retry logic ─────────────────────────────────────────────────────

def test_run_query_retries_on_json_error():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids":       [["pb-001"]],
        "documents": [["doc"]],
        "metadatas": [[{"title": "SSH BF", "severity": "HIGH"}]],
        "distances": [[0.1]],
    }

    bad_response  = {"message": {"content": "not json"}}
    good_response = {"message": {"content": json.dumps(VALID_JSON)}}

    call_count = {"n": 0}
    def side_effect(**kwargs):
        call_count["n"] += 1
        return bad_response if call_count["n"] == 1 else good_response

    with patch.object(rag, "embed", return_value=[0.0] * 768):
        with patch("ollama.chat", side_effect=side_effect):
            result = rag.run_query("test alert", mock_collection, model="llama3")

    assert result is not None
    assert result["attack_type"] == "SSH Brute Force"
    assert call_count["n"] == 2, "Expected exactly 2 LLM calls (1 fail + 1 retry)"