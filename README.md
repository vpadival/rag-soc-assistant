# RAG SOC Assistant

A local, offline **Security Operations Centre (SOC) assistant** powered by Retrieval-Augmented Generation (RAG). Paste a raw alert log or JSON event and the system retrieves relevant playbooks from a vector store, then generates a structured triage report — attack type, severity, MITRE ATT&CK mapping, mitigation steps, and detection recommendations.

Everything runs locally: no API keys, no cloud dependencies.

---

## Architecture

```
Alert (text / JSON)
    │
    ▼
React SOC Dashboard (Vite + CSS Modules)
    │
    ▼
FastAPI REST API
    │
    ├─► Embed (nomic-embed-text via Ollama)
    │       │
    │       ▼
    │   Vector Search (ChromaDB — cosine similarity)
    │       │
    │       ▼
    │   Top-2 Playbooks retrieved
    │
    └─► LLM Generation (llama3 via Ollama)
            │
            ▼
    Structured JSON triage report
```

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Ollama (local) | No API key, fully offline |
| Default model | `llama3` | Swappable via UI dropdown or request body |
| Embedding model | `nomic-embed-text` | Via Ollama, no external dependency |
| Vector store | ChromaDB (persistent) | Zero config, file-based |
| Similarity metric | Cosine | Set at collection creation |
| REST API | FastAPI + Uvicorn | Async, auto-docs, Pydantic v2 validation |
| Frontend | React 18 + Vite | CSS Modules, no UI library |
| Language | Python 3.11+ | — |

---

## Knowledge Base

20 playbooks covering the full MITRE ATT&CK kill chain:

| ID | Title | Severity | MITRE |
|---|---|---|---|
| pb-001 | SSH Brute Force Attack | HIGH | T1110.001 |
| pb-002 | Port Scanning / Reconnaissance | MEDIUM | T1046 |
| pb-003 | Phishing / Credential Harvesting | HIGH | T1566.002 |
| pb-004 | Privilege Escalation Attempt | CRITICAL | T1548 |
| pb-005 | Data Exfiltration | CRITICAL | T1041 |
| pb-006 | Malware / Ransomware Execution | CRITICAL | T1486 |
| pb-007 | Lateral Movement via Pass-the-Hash | CRITICAL | T1550.002 |
| pb-008 | Command and Control (C2) Beacon | CRITICAL | T1071.001 |
| pb-009 | Insider Threat / Abnormal Data Access | HIGH | T1078 |
| pb-010 | DNS Tunneling | HIGH | T1071.004 |
| pb-011 | Web Application SQL Injection | HIGH | T1190 |
| pb-012 | Unauthorized Cloud Storage Access | HIGH | T1530 |
| pb-013 | Account Takeover / Credential Stuffing | HIGH | T1110.004 |
| pb-014 | Cryptomining / Resource Hijacking | MEDIUM | T1496 |
| pb-015 | Zero-Day Exploit Attempt | CRITICAL | T1203 |
| pb-016 | Active Directory Kerberoasting | HIGH | T1558.003 |
| pb-017 | Container Escape Attempt | CRITICAL | T1611 |
| pb-018 | Supply Chain / Dependency Confusion | HIGH | T1195.001 |
| pb-019 | DDoS / Volumetric Attack | HIGH | T1498 |
| pb-020 | Exposed Credentials in Source Code | HIGH | T1552.001 |

---

## Project Structure

```
rag-soc-assistant/
├── data/
│   └── playbooks.json          ← 20 security playbooks (knowledge base)
├── chroma_store/               ← Auto-generated ChromaDB vector store (git-ignored)
├── ingest.py                   ← Embeds playbooks into ChromaDB (run once)
├── rag.py                      ← Core RAG pipeline (embed → retrieve → generate)
├── models.py                   ← Pydantic request/response schemas
├── api.py                      ← FastAPI REST API
├── pyrightconfig.json          ← Pylance/Pyright type checker config
├── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx             ← Shell + routing + state
    │   ├── api.js              ← All fetch calls (API_BASE = localhost:8000)
    │   ├── index.css           ← Global CSS vars + resets
    │   ├── main.jsx
    │   └── components/
    │       ├── Topbar.jsx          ← Status bar (API status, model, KB count)
    │       ├── Sidebar.jsx         ← Severity index, playbook list, history
    │       ├── AlertInput.jsx      ← Alert textarea + model switcher dropdown
    │       ├── TriageReport.jsx    ← Structured triage output with match quality
    │       └── HealthPanel.jsx     ← API health + re-ingest button
    ├── package.json
    └── vite.config.js
```

---

## Setup

### Prerequisites

- Python 3.11 or 3.12 (recommended — avoids numpy build issues on 3.14)
- Node.js 18+
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

Run once, or whenever `data/playbooks.json` is updated.

### 5. Start the API server

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Usage

### React Dashboard

Open **http://localhost:5173** with both `ollama serve` and the FastAPI server running.

- Paste any raw alert log, JSON event, or plain-text description into the input pane
- Select a model from the dropdown (populated from your live Ollama instance)
- Hit **ANALYZE** or press `Ctrl+Enter`
- The triage report renders on the right with severity, MITRE mapping, explanation, and numbered mitigation steps
- Click any playbook in the sidebar to pre-fill a template query
- Query history persists across page refreshes

### CLI — single query

```bash
python rag.py --query "Multiple SSH failures from 45.33.32.156 targeting root"
```

### CLI — interactive REPL

```bash
python rag.py
```

### REST API

Interactive docs: **http://localhost:8000/docs**

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
| 3 | React SOC Dashboard UI | ✅ Complete |
| 4 | Full stack wired together | ✅ Complete |

---

## Configuration

**Swap the LLM model** — select from the dropdown in the UI, or pass `"model"` in the request body:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"alert": "Suspicious outbound transfer detected", "model": "qwen2.5-coder:7b"}'
```

Or use the `--model` flag with the CLI:

```bash
python rag.py --model mistral --query "Suspicious outbound transfer detected"
```

**Add playbooks** — edit `data/playbooks.json` and re-run `python ingest.py` (or call `POST /ingest` with the server running).

Each playbook requires: `id`, `title`, `severity`, `mitre_technique`, `description`, `indicators[]`, `detection_rule`, `response_steps[]`.

---

## Notes

- `chroma_store/` is git-ignored — run `python ingest.py` after cloning
- CORS is open (`allow_origins=["*"]`) for local development — restrict before any deployment
- The frontend connects directly to `http://localhost:8000` — change `API_BASE` in `frontend/src/api.js` if needed
- Playbooks with cosine distance > 0.5 are flagged as weak matches in the UI