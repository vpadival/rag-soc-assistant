"""
rag.py
------
Core RAG pipeline for the SOC Assistant.

Usage:
    # Interactive mode (REPL)
    python rag.py

    # Single query from CLI
    python rag.py --query "Multiple SSH failures from 45.x.x.x"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import chromadb
import ollama
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

CHROMA_PATH: str = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION:  str = "soc_playbooks"
EMBED_MODEL: str = "nomic-embed-text"
TOP_K:       int = 2

# Mutable runtime setting — lowercase so Pylance does not treat it as a constant
_llm_model: str = "llama3"   # change to "mistral" or "phi3" if preferred

# Type alias for a single retrieved hit
Hit = dict[str, Any]


# ── Embedding ────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Return an embedding vector for *text* using the Ollama embedding model."""
    response: dict[str, Any] = ollama.embeddings(model=EMBED_MODEL, prompt=text)  # type: ignore[assignment]
    embedding: list[float] = response["embedding"]
    return embedding


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
) -> list[Hit]:
    """
    Embed *query* and return the top_k most similar playbook chunks.
    Each hit is a dict with keys: id, document, metadata, distance.
    """
    query_vec: list[float] = embed(query)

    results: chromadb.QueryResult = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids:       list[str]            = results["ids"][0]
    documents: list[str]            = results["documents"][0]   # type: ignore[index]
    metadatas: list[dict[str, Any]] = results["metadatas"][0]   # type: ignore[index]
    distances: list[float]          = results["distances"][0]   # type: ignore[index]

    hits: list[Hit] = [
        {
            "id":       ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]
    return hits


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are a Security Operations Centre (SOC) assistant.
You ONLY use the provided CONTEXT documents to answer. Do NOT invent details.
Always respond with a valid JSON object and nothing else — no markdown fences, no preamble.

JSON schema (use exactly these keys):
{
  "attack_type": "<short name of the attack>",
  "severity": "<CRITICAL | HIGH | MEDIUM | LOW>",
  "explanation": "<2-3 sentence explanation of what is happening>",
  "mitigation": ["<step 1>", "<step 2>", ...],
  "detection_recommendation": "<one actionable SIEM/detection rule suggestion>",
  "mitre_attack": "<MITRE ATT&CK technique ID and name, or N/A>"
}"""


def build_prompt(query: str, context_docs: list[Hit]) -> str:
    """Inject retrieved playbook context into the prompt template."""
    context_block: str = "\n\n---\n\n".join(
        f"[Playbook: {h['metadata']['title']} | Severity: {h['metadata']['severity']}]\n{h['document']}"
        for h in context_docs
    )
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"ALERT:\n{query}\n\n"
        "Respond ONLY with a JSON object matching the schema. No other text."
    )


# ── Generation ────────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    """Send the prompt to the local Ollama LLM and return the raw text reply."""
    # stream=False explicitly selects the non-streaming overload so Pylance
    # can resolve the return type to ChatResponse (not Iterator[ChatResponse]).
    response: ollama.ChatResponse = ollama.chat(  # type: ignore[reportUnknownMemberType]
        model=_llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        stream=False,
    )
    content: str = response.message.content or ""
    return content.strip()


def parse_response(raw: str) -> dict[str, Any]:
    """
    Parse the LLM JSON reply.
    Strips accidental markdown fences if the model adds them.
    """
    cleaned: str = raw
    if cleaned.startswith("```"):
        lines: list[str] = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    parsed: dict[str, Any] = json.loads(cleaned)
    return parsed


# ── Display ───────────────────────────────────────────────────────────────────

SEVERITY_COLOR: dict[str, str] = {
    "CRITICAL": "bold red",
    "HIGH":     "bold orange1",
    "MEDIUM":   "bold yellow",
    "LOW":      "bold green",
}


def display_result(query: str, result: dict[str, Any], hits: list[Hit]) -> None:
    """Pretty-print the structured SOC analysis to the terminal."""
    sev:   str = result.get("severity", "UNKNOWN")
    color: str = SEVERITY_COLOR.get(sev, "white")

    console.print()
    console.rule("[cyan]SOC Analysis[/]")
    console.print(Panel(f"[bold]Query:[/] {query}", style="dim", box=box.SIMPLE))

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold cyan", width=28)
    table.add_column("Value", style="white",     ratio=1)

    table.add_row("Attack Type",  str(result.get("attack_type",  "N/A")))
    table.add_row("Severity",     f"[{color}]{sev}[/]")
    table.add_row("MITRE ATT&CK", str(result.get("mitre_attack", "N/A")))
    table.add_row("Explanation",  str(result.get("explanation",  "N/A")))
    console.print(table)

    mitigations: list[str] = result.get("mitigation", [])
    if mitigations:
        console.print("\n[bold cyan]Mitigation Steps:[/]")
        for i, step in enumerate(mitigations, 1):
            console.print(f"  [dim]{i}.[/] {step}")

    det: str = result.get("detection_recommendation", "")
    if det:
        console.print("\n[bold cyan]Detection Recommendation:[/]")
        console.print(Panel(det, style="dim blue", box=box.SIMPLE))

    console.print(f"\n[dim]Retrieved {len(hits)} playbook(s):[/]")
    for h in hits:
        title:    str   = h["metadata"].get("title", "unknown")
        distance: float = h["distance"]
        console.print(f"  [dim]· {title} (distance: {distance:.4f})[/]")

    console.rule()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    collection: chromadb.Collection,
) -> Optional[dict[str, Any]]:
    """
    Full RAG pipeline: embed query → retrieve → build prompt → generate → parse.
    Returns the structured result dict, or None on error.
    """
    raw: str = ""
    try:
        with console.status("[cyan]Retrieving relevant playbooks…[/]"):
            hits: list[Hit] = retrieve(query, collection)

        prompt: str = build_prompt(query, hits)

        with console.status(f"[cyan]Generating analysis with {_llm_model}…[/]"):
            raw = generate(prompt)

        result: dict[str, Any] = parse_response(raw)
        display_result(query, result, hits)
        return result

    except json.JSONDecodeError as exc:
        console.print(f"[red]⚠ JSON parse error:[/] {exc}")
        console.print(f"[dim]Raw LLM output:[/]\n{raw}")
        return None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]⚠ Error:[/] {exc}")
        return None


def interactive_loop(collection: chromadb.Collection) -> None:
    """Read-eval-print loop for interactive SOC queries."""
    console.print(Panel(
        "[bold cyan]SOC Assistant[/] — RAG Mode\n"
        "[dim]Type your alert and press Enter. Type [bold]exit[/] to quit.[/]",
        box=box.ROUNDED,
    ))
    while True:
        try:
            query: str = console.input("\n[bold green]Alert>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break
        run_query(query, collection)

    console.print("\n[dim]Goodbye.[/]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # _llm_model is lowercase so it is not treated as a module-level constant.
    # We still need `global` to reassign it from inside main().
    global _llm_model

    parser = argparse.ArgumentParser(description="SOC Assistant — RAG pipeline")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Run a single alert query and exit")
    parser.add_argument("--model", "-m", type=str, default=_llm_model,
                        help=f"Ollama LLM model to use (default: {_llm_model})")
    args = parser.parse_args()
    _llm_model = args.model

    if not os.path.exists(CHROMA_PATH):
        console.print(
            "[red]✗ ChromaDB store not found.[/] "
            "Run [bold]python ingest.py[/] first."
        )
        sys.exit(1)

    # chromadb.PersistentClient returns a chromadb.Client — no need to
    # reference the private ClientAPI type at all.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection: chromadb.Collection = client.get_collection(COLLECTION)

    if args.query:
        run_query(args.query, collection)
    else:
        interactive_loop(collection)


if __name__ == "__main__":
    main()