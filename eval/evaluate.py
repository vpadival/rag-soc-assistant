"""
eval/evaluate.py
----------------
Retrieval evaluation for the SOC Assistant RAG pipeline.

Metrics computed:
  Precision@K  — fraction of retrieved playbooks that are relevant
  Recall@K     — fraction of relevant playbooks that were retrieved
  MRR          — mean reciprocal rank of first relevant hit
  Hit Rate@K   — fraction of queries where at least one relevant hit retrieved

Usage:
    # Ensure chroma_store exists (run python ingest.py first)
    python eval/evaluate.py
    python eval/evaluate.py --top-k 4
    python eval/evaluate.py --top-k 1 --top-k 2 --top-k 4   # compare K values
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb

import rag

LABELS_PATH = ROOT / "eval" / "labelled_alerts.json"


# ── Metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


# ── Severity accuracy ─────────────────────────────────────────────────────────

def severity_match(retrieved_ids: list[str], relevant_ids: list[str],
                   collection: chromadb.Collection) -> bool | None:
    """Check whether the top retrieved hit has the same severity as the ground truth."""
    if not retrieved_ids or not relevant_ids:
        return None
    # Fetch severity metadata for the top retrieved hit
    try:
        res = collection.get(ids=[retrieved_ids[0]], include=["metadatas"])
        ret_sev = res["metadatas"][0].get("severity", "") if res["metadatas"] else ""
        gt_res  = collection.get(ids=[relevant_ids[0]], include=["metadatas"])
        gt_sev  = gt_res["metadatas"][0].get("severity", "") if gt_res["metadatas"] else ""
        return ret_sev == gt_sev
    except Exception:
        return None


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(top_k: int, collection: chromadb.Collection,
             labels: list[dict[str, Any]]) -> dict[str, float]:
    precisions: list[float] = []
    recalls:    list[float] = []
    rrs:        list[float] = []
    hit_flags:  list[int]   = []
    sev_flags:  list[int]   = []

    for sample in labels:
        alert    = sample["alert"]
        relevant = sample["ground_truth_playbook_ids"]

        hits     = rag.retrieve(alert, collection, top_k=top_k)
        ret_ids  = [h["id"] for h in hits]

        p = precision_at_k(ret_ids, relevant)
        r = recall_at_k(ret_ids, relevant)
        rr = reciprocal_rank(ret_ids, relevant)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        hit_flags.append(1 if rr > 0 else 0)

        sev = severity_match(ret_ids, relevant, collection)
        if sev is not None:
            sev_flags.append(1 if sev else 0)

        print(
            f"  [{sample['id']}] P@{top_k}={p:.2f}  R@{top_k}={r:.2f}  RR={rr:.2f}  "
            f"retrieved={ret_ids}"
        )

    n = len(labels)
    return {
        f"P@{top_k}":        sum(precisions) / n,
        f"R@{top_k}":        sum(recalls) / n,
        "MRR":               sum(rrs) / n,
        f"HitRate@{top_k}":  sum(hit_flags) / n,
        "SeverityAcc":       sum(sev_flags) / len(sev_flags) if sev_flags else float("nan"),
    }


def print_results_table(results: dict[int, dict[str, float]]) -> None:
    header_keys = sorted({k for r in results.values() for k in r})
    print("\n" + "=" * 72)
    print(f"{'Metric':<18}", end="")
    for k in sorted(results):
        print(f"  top_k={k:<5}", end="")
    print()
    print("-" * 72)

    shown: set[str] = set()
    for k in sorted(results):
        for metric in results[k]:
            if metric not in shown:
                shown.add(metric)
                print(f"{metric:<18}", end="")
                for kk in sorted(results):
                    val = results[kk].get(metric, float("nan"))
                    print(f"  {val:>9.4f} ", end="")
                print()
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument(
        "--top-k", "-k", type=int, action="append", dest="top_ks", default=None,
        help="TOP_K to evaluate (repeat for multiple: -k 1 -k 2 -k 4)"
    )
    args = parser.parse_args()
    top_ks: list[int] = sorted(set(args.top_ks)) if args.top_ks else [1, 2, 4]

    chroma_path = str(ROOT / "chroma_store")
    if not os.path.exists(chroma_path):
        print("✗ chroma_store not found. Run: python ingest.py")
        sys.exit(1)

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels: list[dict[str, Any]] = json.load(f)

    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(rag.COLLECTION)

    all_results: dict[int, dict[str, float]] = {}

    for k in top_ks:
        print(f"\n── Evaluating TOP_K = {k} ({'─' * 50})")
        all_results[k] = evaluate(k, collection, labels)

    print_results_table(all_results)

    # Write JSON report
    report_path = ROOT / "eval" / "results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()