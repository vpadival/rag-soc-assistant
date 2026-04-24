"""
models.py
---------
Pydantic request / response schemas for the SOC Assistant REST API.
Kept separate from api.py so they can be imported by tests or future clients.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Request bodies ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    alert: str = Field(
        ...,
        min_length=1,
        description="Raw alert string to analyse (e.g. 'Multiple SSH failures from 45.33.32.156')",
        examples=["Multiple failed SSH logins from 192.168.1.200 targeting root account"],
    )
    model: str = Field(
        default="llama3",
        description="Ollama LLM model to use for generation",
        examples=["llama3", "mistral", "phi3"],
    )
    top_k: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Number of playbooks to retrieve (1–6)",
    )


class IngestRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="If true, re-embeds all playbooks even if the store already exists",
    )


# ── Sub-models ─────────────────────────────────────────────────────────────────

class RetrievedPlaybook(BaseModel):
    id: str
    title: str
    severity: str
    distance: float = Field(description="Cosine distance — lower = more similar")


class AnalysisResult(BaseModel):
    attack_type: str
    severity: str
    explanation: str
    mitigation: list[str]
    detection_recommendation: str
    mitre_attack: str


# ── Response bodies ────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    alert: str
    model_used: str
    retrieved_playbooks: list[RetrievedPlaybook]
    analysis: AnalysisResult


class PlaybookEntry(BaseModel):
    """
    Flexible schema that handles both old and new playbook field names.
    All fields beyond id/title/severity are optional so adding new playbooks
    with different field names never causes a 422 on GET /playbooks.
    """
    id: str
    title: str
    severity: str

    # MITRE — old schema: mitre_attack, new schema: mitre_technique
    mitre_attack: str | None = None
    mitre_technique: str | None = None
    mitre_tactic: str | None = None

    # Explanation — old schema: explanation, new schema: description
    explanation: str | None = None
    description: str | None = None

    # Detection — old schema: detection, new schema: detection_rule
    detection: str | None = None
    detection_rule: str | None = None

    # Steps — old schema: mitigation, new schema: response_steps
    mitigation: list[str] = []
    response_steps: list[str] = []

    # Shared optional fields
    tags: list[str] = []
    indicators: list[str] = []

    model_config = {"extra": "allow"}  # pass through any unknown fields silently


class PlaybooksResponse(BaseModel):
    count: int
    playbooks: list[PlaybookEntry]


class OllamaStatus(BaseModel):
    reachable: bool
    available_models: list[str]
    required_models_present: bool


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "error"
    chroma_store_ready: bool
    ollama: OllamaStatus
    playbook_count: int


class IngestResponse(BaseModel):
    success: bool
    playbooks_embedded: int
    skipped: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
    raw_llm_output: str | None = None