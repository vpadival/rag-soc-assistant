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
import re
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
DEFAULT_LLM: str = "llama3"

# Cosine distance above which a retrieved playbook is considered a weak /
# unreliable match. The frontend already shows this to the user as a cosmetic
# "weak match" badge (frontend/src/components/TriageReport.jsx), but nothing
# on the backend used it to change behavior — the LLM was always asked to
# produce a confident-sounding structured report even off a bad match.
# Callers (api.py) should check this before generating.
DISTANCE_THRESHOLD: float = 0.5

# Sampling temperature for generation. This is a fact-constrained structured
# extraction task (attack classification, MITRE mapping), not open-ended
# writing, so low temperature is preferred to reduce free invention.
DEFAULT_TEMPERATURE: float = 0.2

# Type alias for a single retrieved hit
Hit = dict[str, Any]


# ── Embedding ────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Return an embedding vector for *text* using the Ollama embedding model."""
    response: dict[str, Any] = ollama.embeddings(model=EMBED_MODEL, prompt=text)  # type: ignore[assignment]
    return response["embedding"]


# ── Retrieval ─────────────────────────────────────────────────────────────────

# Common English words that carry no topical signal for overlap purposes.
# Deliberately small and conservative — this is a cheap supplementary check,
# not a real relevance model. See has_reliable_match() docstring for why it
# exists alongside (not instead of) the distance threshold.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "was", "were", "been", "being", "has", "have",
    "had", "not", "but", "with", "from", "this", "that", "these", "those",
    "their", "they", "them", "than", "then", "there", "here", "who", "what",
    "when", "where", "why", "how", "all", "any", "some", "into", "onto",
    "over", "under", "about", "after", "before", "during", "while", "could",
    "would", "should", "will", "shall", "may", "might", "must", "can", "did",
    "does", "doing", "each", "few", "more", "most", "other", "such", "only",
    "own", "same", "out", "off", "again", "further", "once", "possibly",
    "occasionally", "reported", "keeps",
})

# Minimum number of shared, non-generic tokens the alert must have with its
# best-matched playbook's text for the match to count as reliable.
MIN_LEXICAL_OVERLAP: int = 1


def _significant_tokens(text: str) -> set[str]:
    """Lowercase alnum tokens of length >= 3, minus stopwords."""
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def match_diagnostics(
    query: str,
    hits: list[Hit],
    threshold: float = DISTANCE_THRESHOLD,
    min_overlap: int = MIN_LEXICAL_OVERLAP,
) -> dict[str, Any]:
    """
    Like has_reliable_match(), but returns *why* — so callers building a
    user-facing explanation can say what actually failed instead of always
    blaming distance regardless of which check rejected the match.

    Returns a dict with:
      - reliable: bool
      - best_distance: float | None (None if there were no hits at all)
      - reason: "no_hits" | "distance" | "overlap" | "" (empty when reliable)
    """
    if not hits:
        return {"reliable": False, "best_distance": None, "reason": "no_hits"}

    best = min(hits, key=lambda h: h["distance"])
    best_distance = best["distance"]

    if best_distance > threshold:
        return {"reliable": False, "best_distance": best_distance, "reason": "distance"}

    query_tokens = _significant_tokens(query)
    doc_tokens = _significant_tokens(
        best.get("document", "") + " " + best["metadata"].get("title", "")
    )
    if len(query_tokens & doc_tokens) < min_overlap:
        return {"reliable": False, "best_distance": best_distance, "reason": "overlap"}

    return {"reliable": True, "best_distance": best_distance, "reason": ""}


def has_reliable_match(
    query: str,
    hits: list[Hit],
    threshold: float = DISTANCE_THRESHOLD,
    min_overlap: int = MIN_LEXICAL_OVERLAP,
) -> bool:
    """
    True if the best retrieved hit is close enough AND topically overlapping
    enough to trust as grounding for generation.

    Distance alone is not sufficient here: cosine distance from
    nomic-embed-text over this playbook set does not reliably separate
    genuinely relevant alerts from unrelated text — an alert describing an
    office chair squeaking was observed to score 0.47 distance (under the
    0.5 threshold) against an unrelated security playbook. Bi-encoder
    embedding spaces like this one are often anisotropic enough that
    "unrelated" text still lands at a moderate rather than large distance.

    As a cheap supplementary check (no new dependency, unlike a proper
    cross-encoder reranker — see eval/reranker.py for that heavier, more
    reliable option), this also requires the alert text to share at least
    `min_overlap` non-generic token(s) with the best-matched playbook's own
    text. This is a coarse heuristic, not a relevance model: it can still be
    fooled by incidental keyword overlap, and it can reject a genuine match
    phrased with unusual vocabulary. It narrows the specific failure mode
    observed (topically unrelated text passing on distance alone); it does
    not replace proper reranking.

    See match_diagnostics() for a version that also reports which check
    failed, rather than just true/false.
    """
    return match_diagnostics(query, hits, threshold, min_overlap)["reliable"]


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

    return [
        {
            "id":       ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]


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

RETRY_SYSTEM_PROMPT: str = SYSTEM_PROMPT + """

IMPORTANT: Your previous response was not valid JSON. Return ONLY the JSON object.
No explanation, no markdown, no backticks — just the raw JSON object starting with '{'.
"""


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

def generate(
    prompt: str,
    model: str = DEFAULT_LLM,
    system: str = SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Send the prompt to the local Ollama LLM and return the raw text reply.
    model is passed explicitly — no global mutation.
    temperature defaults low (see DEFAULT_TEMPERATURE) since this is a
    fact-constrained extraction task, not open-ended writing.
    """
    response: ollama.ChatResponse = ollama.chat(  # type: ignore[reportUnknownMemberType]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": temperature},
        stream=False,
    )
    if isinstance(response, dict):
        content: str = response.get("message", {}).get("content", "") or ""
    else:
        content = response.message.content or ""
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
    # Trim any leading non-JSON text to the first '{'
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    return json.loads(cleaned)


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
    model: str = DEFAULT_LLM,
    max_retries: int = 2,
) -> Optional[dict[str, Any]]:
    """
    Full RAG pipeline: embed query → retrieve → build prompt → generate → parse.
    Retries once with a stricter system prompt on JSON parse failure.
    Returns the structured result dict, or None on error.
    model is passed explicitly — no global state mutation.
    """
    raw: str = ""
    try:
        with console.status("[cyan]Retrieving relevant playbooks…[/]"):
            hits: list[Hit] = retrieve(query, collection)

        prompt: str = build_prompt(query, hits)

        for attempt in range(max_retries):
            sys_prompt = SYSTEM_PROMPT if attempt == 0 else RETRY_SYSTEM_PROMPT
            with console.status(f"[cyan]Generating analysis with {model}… (attempt {attempt + 1})[/]"):
                raw = generate(prompt, model=model, system=sys_prompt)
            try:
                result: dict[str, Any] = parse_response(raw)
                display_result(query, result, hits)
                return result
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    console.print(f"[yellow]⚠ JSON parse failed on attempt {attempt + 1}, retrying…[/]")
                else:
                    console.print(f"[red]⚠ JSON parse error after {max_retries} attempts.[/]")
                    console.print(f"[dim]Raw LLM output:[/]\n{raw}")
                    return None

    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]⚠ Error:[/] {exc}")
        return None


def interactive_loop(collection: chromadb.Collection, model: str = DEFAULT_LLM) -> None:
    """Read-eval-print loop for interactive SOC queries."""
    console.print(Panel(
        f"[bold cyan]SOC Assistant[/] — RAG Mode | model: [bold]{model}[/]\n"
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
        run_query(query, collection, model=model)

    console.print("\n[dim]Goodbye.[/]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SOC Assistant — RAG pipeline")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Run a single alert query and exit")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_LLM,
                        help=f"Ollama LLM model to use (default: {DEFAULT_LLM})")
    args = parser.parse_args()

    if not os.path.exists(CHROMA_PATH):
        console.print(
            "[red]✗ ChromaDB store not found.[/] "
            "Run [bold]python ingest.py[/] first."
        )
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection: chromadb.Collection = client.get_collection(COLLECTION)

    if args.query:
        run_query(args.query, collection, model=args.model)
    else:
        interactive_loop(collection, model=args.model)


if __name__ == "__main__":
    main()