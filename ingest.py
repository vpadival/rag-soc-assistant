"""
ingest.py
---------
Reads data/playbooks.json, embeds each document using Ollama's
nomic-embed-text model, and stores them in a local ChromaDB collection.

Run once (or whenever playbooks.json is updated):
    python ingest.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import chromadb
import ollama
from rich.console import Console
from rich.progress import track

console = Console()

PLAYBOOKS_PATH: str = os.path.join(os.path.dirname(__file__), "data", "playbooks.json")
CHROMA_PATH:    str = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION:     str = "soc_playbooks"
EMBED_MODEL:    str = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text
OLLAMA_TIMEOUT_SEC: float = 5.0

# Use an explicit client so requests fail fast when Ollama is down.
OLLAMA_CLIENT = ollama.Client(timeout=OLLAMA_TIMEOUT_SEC)


def build_document_text(pb: dict[str, Any]) -> str:
    """
    Flatten a playbook dict into a single string for embedding.
    Richer text → better retrieval accuracy.
    """
    indicators: list[str] = pb.get("indicators", [])
    mitigation: list[str] = pb.get("mitigation", [])

    lines: list[str] = [
        f"Title: {pb['title']}",
        f"Severity: {pb['severity']}",
        f"Tags: {', '.join(pb['tags'])}",
        f"MITRE ATT&CK: {pb.get('mitre_attack', 'N/A')}",
        f"Explanation: {pb['explanation']}",
        f"Indicators: {'; '.join(indicators)}",
        f"Mitigation: {'; '.join(mitigation)}",
        f"Detection: {pb['detection']}",
    ]
    return "\n".join(lines)


def embed(text: str) -> list[float]:
    """Return an embedding vector from Ollama."""
    response: dict[str, Any] = OLLAMA_CLIENT.embeddings(model=EMBED_MODEL, prompt=text)  # type: ignore[assignment]
    embedding: list[float] = response["embedding"]
    return embedding


def ollama_is_available() -> bool:
    """Check whether Ollama server is reachable before embedding."""
    try:
        OLLAMA_CLIENT.list()
        return True
    except KeyboardInterrupt:
        console.print("\n[yellow]! Cancelled by user.[/]")
        return False
    except Exception as exc:
        console.print("[bold red]✗ Ollama is not reachable.[/]")
        console.print("  Start Ollama first (for example: [bold]ollama serve[/]).")
        console.print(f"  Timeout used: {OLLAMA_TIMEOUT_SEC}s")
        console.print(f"  Details: {exc}\n")
        return False


def main() -> None:
    console.rule("[bold cyan]SOC Assistant — Knowledge Base Ingestion[/]")

    if not ollama_is_available():
        return

    # ── Load playbooks ───────────────────────────────────────────────────────
    with open(PLAYBOOKS_PATH, "r", encoding="utf-8") as f:
        playbooks: list[dict[str, Any]] = json.load(f)
    console.print(f"[green]✓[/] Loaded [bold]{len(playbooks)}[/] playbooks from {PLAYBOOKS_PATH}")

    # ── Init ChromaDB (persistent, local) ───────────────────────────────────
    # PersistentClient return type is inferred — no need for the private ClientAPI annotation.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection: chromadb.Collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    console.print(f"[green]✓[/] ChromaDB collection '{COLLECTION}' ready at {CHROMA_PATH}")

    # ── Build lists to upsert ────────────────────────────────────────────────
    ids:       list[str]  = []
    documents: list[str]  = []
    # Use Any so the Sequence[Metadata] mismatch is suppressed at the source.
    embeddings: list[Any] = []
    metadatas:  list[Any] = []

    failed_ids: list[str] = []

    for pb in track(playbooks, description="Embedding playbooks…"):
        doc_text: str = build_document_text(pb)

        try:
            vec: list[float] = embed(doc_text)
        except Exception as exc:
            pb_id: str = pb.get("id", "unknown")
            failed_ids.append(pb_id)
            console.print(f"[yellow]![/] Skipping playbook id='{pb_id}' due to embedding error: {exc}")
            continue

        ids.append(pb["id"])
        embeddings.append(vec)
        documents.append(doc_text)
        metadatas.append({
            "title":        pb["title"],
            "severity":     pb["severity"],
            "tags":         ", ".join(pb["tags"]),
            "mitre_attack": pb.get("mitre_attack", ""),
        })

    if not ids:
        console.print("\n[bold yellow]! No embeddings were generated. Nothing was written to ChromaDB.[/]")
        return

    # upsert = insert or update — safe to run multiple times
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    console.print(f"\n[bold green]✓ Done![/] {len(ids)} playbooks embedded and stored.")
    console.print(f"  Collection count: {collection.count()} documents\n")

    if failed_ids:
        console.print(f"[yellow]! Skipped {len(failed_ids)} playbooks: {', '.join(failed_ids)}[/]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]! Ingestion cancelled by user.[/]")