"""
api.py
------
FastAPI REST API wrapping the SOC Assistant RAG pipeline.

Start the server:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /analyze          — run the full RAG pipeline on an alert string
    GET  /health           — server, Ollama, and ChromaDB status
    GET  /playbooks        — list all playbooks in the knowledge base
    POST /ingest           — re-embed playbooks into ChromaDB on demand
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

# Ensure the project root is on sys.path so the uvicorn reloader subprocess
# can find the local modules (rag, ingest, models) regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import chromadb
import ollama
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Internal modules
import rag  # reuses embed / retrieve / build_prompt / generate / parse_response
from ingest import (
    PLAYBOOKS_PATH,
    OLLAMA_CLIENT,
    build_document_text,
    embed as ingest_embed,
)
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    OllamaStatus,
    PlaybookEntry,
    PlaybooksResponse,
    RetrievedPlaybook,
)

# ── Paths / constants ─────────────────────────────────────────────────────────

CHROMA_PATH: str = rag.CHROMA_PATH
COLLECTION:  str = rag.COLLECTION
REQUIRED_MODELS: list[str] = ["llama3", "nomic-embed-text"]


# ── App-wide shared state (initialised once at startup) ───────────────────────

class AppState:
    chroma_client: Any | None = None
    collection:    chromadb.Collection | None = None


_state = AppState()


# ── Lifespan: init ChromaDB once at startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup. Initialises the persistent ChromaDB client and
    holds the collection reference for the lifetime of the server process.
    All request handlers share this single instance.
    """
    if os.path.exists(CHROMA_PATH):
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _state.chroma_client = _client
        _state.collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    # else: collection stays None — /health will report degraded state

    yield  # server is running

    # Teardown (nothing needed for ChromaDB — it auto-persists)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SOC Assistant API",
    description=(
        "REST API for the RAG-powered Security Operations Centre assistant. "
        "Wraps a local Ollama LLM + ChromaDB vector store to analyse security alerts "
        "against a curated playbook knowledge base."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow all origins for local / dashboard use — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_collection() -> chromadb.Collection:
    """Return the shared ChromaDB collection or raise 503."""
    if _state.collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB store not initialised. "
                "Run POST /ingest to embed playbooks first."
            ),
        )
    return _state.collection


def _ollama_status() -> OllamaStatus:
    """Probe Ollama and return a structured status object."""
    try:
        models_response = OLLAMA_CLIENT.list()
        # .models is a list[ModelResponse]; each has a .model attribute
        available: list[str] = [m.model for m in models_response.models if m.model is not None]
        required_present = all(
            any(a.startswith(req) for a in available)
            for req in REQUIRED_MODELS
        )
        return OllamaStatus(
            reachable=True,
            available_models=available,
            required_models_present=required_present,
        )
    except Exception:
        return OllamaStatus(
            reachable=False,
            available_models=[],
            required_models_present=False,
        )


def _load_playbooks() -> list[dict[str, Any]]:
    """Load playbooks.json from disk."""
    with open(PLAYBOOKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── POST /analyze ─────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        422: {"model": ErrorResponse, "description": "LLM returned unparseable JSON"},
        503: {"model": ErrorResponse, "description": "Ollama or ChromaDB unavailable"},
    },
    summary="Analyse a security alert",
    tags=["RAG Pipeline"],
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full RAG pipeline:

    1. Embed the alert string via `nomic-embed-text`
    2. Retrieve the top-K most similar playbooks from ChromaDB
    3. Build a prompt injecting retrieved context
    4. Send to the local Ollama LLM
    5. Parse and return structured JSON analysis
    """
    collection = _require_collection()

    # Override the module-level model if the caller requested a different one
    original_model = rag.get_llm_model()
    rag.set_llm_model(request.model)

    raw_output: str = ""
    try:
        # Retrieve
        hits: list[rag.Hit] = rag.retrieve(request.alert, collection, top_k=request.top_k)

        # Build prompt & generate
        prompt: str = rag.build_prompt(request.alert, hits)
        raw_output = rag.generate(prompt)

        # Parse
        parsed: dict[str, Any] = rag.parse_response(raw_output)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"LLM returned unparseable JSON: {exc}. Raw output: {raw_output!r}",
        ) from exc
    except ollama.ResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        rag.set_llm_model(original_model)  # always restore

    # Build typed response
    retrieved = [
        RetrievedPlaybook(
            id=h["id"],
            title=h["metadata"].get("title", ""),
            severity=h["metadata"].get("severity", ""),
            distance=round(h["distance"], 6),
        )
        for h in hits
    ]

    analysis = AnalysisResult(
        attack_type=parsed.get("attack_type", "Unknown"),
        severity=parsed.get("severity", "UNKNOWN"),
        explanation=parsed.get("explanation", ""),
        mitigation=parsed.get("mitigation", []),
        detection_recommendation=parsed.get("detection_recommendation", ""),
        mitre_attack=parsed.get("mitre_attack", "N/A"),
    )

    return AnalyzeResponse(
        alert=request.alert,
        model_used=request.model,
        retrieved_playbooks=retrieved,
        analysis=analysis,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server and dependency health check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Returns the health of all system components:

    - ChromaDB store presence and document count
    - Ollama reachability and required model availability
    """
    ollama_stat = _ollama_status()
    chroma_ready = _state.collection is not None
    _col = _state.collection
    playbook_count = _col.count() if _col is not None else 0

    all_good = chroma_ready and ollama_stat.reachable and ollama_stat.required_models_present
    degraded = chroma_ready or ollama_stat.reachable  # at least something is up

    return HealthResponse(
        status="ok" if all_good else ("degraded" if degraded else "error"),
        chroma_store_ready=chroma_ready,
        ollama=ollama_stat,
        playbook_count=playbook_count,
    )


# ── GET /playbooks ────────────────────────────────────────────────────────────

@app.get(
    "/playbooks",
    response_model=PlaybooksResponse,
    summary="List all playbooks in the knowledge base",
    tags=["Knowledge Base"],
)
async def list_playbooks() -> PlaybooksResponse:
    """
    Returns the full contents of `data/playbooks.json` — the static knowledge
    base used to build the ChromaDB vector store.
    """
    try:
        raw: list[dict[str, Any]] = _load_playbooks()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playbooks file not found at {PLAYBOOKS_PATH}",
        )

    playbooks = [
        PlaybookEntry(
            id=pb["id"],
            title=pb["title"],
            severity=pb["severity"],
            tags=pb.get("tags", []),
            mitre_attack=pb.get("mitre_attack", "N/A"),
            explanation=pb.get("explanation", ""),
            indicators=pb.get("indicators", []),
            mitigation=pb.get("mitigation", []),
            detection=pb.get("detection", ""),
        )
        for pb in raw
    ]

    return PlaybooksResponse(count=len(playbooks), playbooks=playbooks)


# ── POST /ingest ──────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Ollama unavailable"},
    },
    summary="Re-embed playbooks into ChromaDB",
    tags=["Operations"],
)
async def ingest(request: IngestRequest = IngestRequest()) -> IngestResponse:
    """
    Reads `data/playbooks.json`, embeds every playbook using `nomic-embed-text`,
    and upserts them into ChromaDB. Safe to call multiple times (upsert = idempotent).

    Use `force=true` to re-embed even if the store already contains documents.
    The in-process collection reference is refreshed automatically after ingestion.
    """
    # Soft-guard: if store exists and not forced, return early
    if not request.force and _state.collection is not None:
        current_count = _state.collection.count()
        if current_count > 0:
            return IngestResponse(
                success=True,
                playbooks_embedded=current_count,
                skipped=0,
                message=(
                    f"Store already contains {current_count} documents. "
                    "Pass force=true to re-embed."
                ),
            )

    # Verify Ollama is up before doing anything expensive
    ollama_stat = _ollama_status()
    if not ollama_stat.reachable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama is not reachable. Start Ollama before ingesting.",
        )

    # Load playbooks
    try:
        playbooks: list[dict[str, Any]] = _load_playbooks()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playbooks file not found at {PLAYBOOKS_PATH}",
        )

    # (Re-)initialise ChromaDB client + collection
    _new_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _state.chroma_client = _new_client
    collection: chromadb.Collection = _new_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids:        list[str] = []
    documents:  list[str] = []
    embeddings: list[Any] = []
    metadatas:  list[Any] = []
    skipped:    int       = 0

    for pb in playbooks:
        doc_text: str = build_document_text(pb)
        try:
            vec: list[float] = ingest_embed(doc_text)
        except Exception:
            skipped += 1
            continue

        ids.append(pb["id"])
        embeddings.append(vec)
        documents.append(doc_text)
        metadatas.append({
            "title":        pb["title"],
            "severity":     pb["severity"],
            "tags":         ", ".join(pb.get("tags", [])),
            "mitre_attack": pb.get("mitre_attack", ""),
        })

    if not ids:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No embeddings were generated. Check that nomic-embed-text is available.",
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    # Update the shared state so subsequent /analyze calls use the fresh collection
    _state.collection = collection

    return IngestResponse(
        success=True,
        playbooks_embedded=len(ids),
        skipped=skipped,
        message=f"Successfully embedded {len(ids)} playbooks. {skipped} skipped.",
    )


# ── Root redirect to docs ─────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")