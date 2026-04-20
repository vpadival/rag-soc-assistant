# RAG SOC Assistant

> 🚧 **Work in Progress (WIP):** This repository is still under active development. Features, APIs, and project structure may change frequently.

A local, offline **Security Operations Centre (SOC) assistant** powered by Retrieval-Augmented Generation (RAG). It analyses raw security alerts against a curated playbook knowledge base and returns structured triage output — attack type, severity, MITRE ATT&CK mapping, mitigation steps, and detection recommendations.

Everything runs locally: no API keys, no cloud dependencies.

---

## Architecture

```
Alert (text)
    │
    ▼
Embed (nomic-embed-text via Ollama)
    │
    ▼
Vector Search (ChromaDB — cosine similarity)
    │
    ▼
Top-K Playbooks retrieved
    │
    ▼
Prompt built (system + context + alert)
    │
    ▼
LLM Generation (llama3 via Ollama)
    │
    ▼
Structured JSON response
```

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Ollama (local) | No API key, fully offline |
| Default model | `llama3` | Swappable via flag / request body |
| Embedding model | `nomic-embed-text` | Via Ollama, no external dependency |
| Vector store | ChromaDB (persistent) | Zero config, file-based |
| Similarity metric | Cosine | Set at collection creation |
| REST API | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Language | Python 3.11+ | — |

---

## Knowledge Base

6 playbooks covering:

- SSH Brute Force Attack
- Port Scanning / Reconnaissance
- Phishing / Credential Harvesting
- Privilege Escalation Attempt
- Data Exfiltration
- Malware / Ransomware Execution

---

## Project Structure

```
rag-soc-assistant/
├── data/
│   └── playbooks.json       ← Security playbook knowledge base
├── chroma_store/            ← Auto-generated ChromaDB vector store (git-ignored)
├── ingest.py                ← Embeds playbooks into ChromaDB (run once)
├── rag.py                   ← Core RAG pipeline (embed → retrieve → generate)
├── models.py                ← Pydantic request/response schemas
├── api.py                   ← FastAPI REST API (Phase 2)
├── pyrightconfig.json       ← Pylance/Pyright type checker config
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11 or 3.12 (recommended — avoids numpy build issues on 3.14)
- [Ollama](https://ollama.com) installed and running

### 1. Clone and create virtual environment

```bash
git clone https://github.com/vpadival/rag-soc-assistant.git
cd rag-soc-assistant
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull required Ollama models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 4. Ingest playbooks into ChromaDB

```bash
python ingest.py
```

This only needs to be run once (or when `data/playbooks.json` is updated).

---

## Usage

### CLI — single query

```bash
python rag.py --query "Multiple SSH failures from 45.33.32.156 targeting root"
```

### CLI — interactive REPL

```bash
python rag.py
```

### REST API

Start the server:

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at **http://localhost:8000/docs**

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Run RAG pipeline on an alert string |
| `GET` | `/health` | Check Ollama + ChromaDB status |
| `GET` | `/playbooks` | List all playbooks in the knowledge base |
| `POST` | `/ingest` | Re-embed playbooks into ChromaDB on demand |

#### Example — analyze an alert

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"alert": "Ransomware detected on workstation — mass file rename events observed"}'
```

**Response:**

```json
{
  "alert": "Ransomware detected on workstation — mass file rename events observed",
  "model_used": "llama3",
  "retrieved_playbooks": [
    { "id": "pb-006", "title": "Malware / Ransomware Execution", "severity": "CRITICAL", "distance": 0.084 }
  ],
  "analysis": {
    "attack_type": "Ransomware",
    "severity": "CRITICAL",
    "explanation": "Mass file rename events indicate active ransomware encryption...",
    "mitigation": ["Immediately isolate infected host from network", "..."],
    "detection_recommendation": "Alert on >100 file rename events per minute from single process",
    "mitre_attack": "T1486 - Data Encrypted for Impact"
  }
}
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | RAG pipeline (CLI) | ✅ Complete |
| 2 | FastAPI REST API | ✅ Complete |
| 3 | React SOC Dashboard UI | 🔜 Next |
| 4 | Full stack wired together | ⏳ Planned |

---

## Configuration

**Swap the LLM model** — pass `"model": "mistral"` in the `/analyze` request body, or use the `--model` flag with the CLI:

```bash
python rag.py --model mistral --query "Suspicious outbound transfer detected"
```

**Add playbooks** — edit `data/playbooks.json` and re-run `python ingest.py` (or call `POST /ingest` with the server running).