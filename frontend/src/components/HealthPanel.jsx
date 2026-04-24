import { useState } from 'react'
import styles from './HealthPanel.module.css'

export default function HealthPanel({ health, apiStatus, onRecheck, onIngest }) {
  const [ingestMsg, setIngestMsg] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [checking,  setChecking]  = useState(false)

  async function handleRecheck() {
    setChecking(true)
    await onRecheck()
    setChecking(false)
  }

  async function handleIngest() {
    setIngesting(true); setIngestMsg('')
    try {
      const d = await onIngest()
      setIngestMsg(d.message || 'Ingestion complete.')
    } catch {
      setIngestMsg('Failed — is the API running?')
    }
    setIngesting(false)
  }

  // Actual API shape: { status, chroma_store_ready, ollama: { reachable, available_models, required_models_present }, playbook_count }
  const ollama  = health?.ollama || {}
  const models  = ollama.available_models || []

  return (
    <div className={styles.panel}>
      <div className={styles.title}>API Health Report</div>

      {apiStatus === 'offline' ? (
        <div className={styles.errorBox}>
          API unreachable at <code>http://localhost:8000</code>.<br /><br />
          Start your FastAPI server:<br />
          <code>python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000</code>
        </div>
      ) : (
        <>
          {/* Core status */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>endpoint status</div>
            <div className={styles.rows}>
              <Row label="status">
                <Val val={health?.status} good="ok" />
              </Row>
              <Row label="ollama reachable">
                <Bool val={ollama.reachable} />
              </Row>
              <Row label="required models">
                <Bool val={ollama.required_models_present} />
              </Row>
              <Row label="vector store">
                <Bool val={health?.chroma_store_ready} />
              </Row>
              <Row label="playbooks indexed">
                <span>{health?.playbook_count ?? '—'}</span>
              </Row>
            </div>
          </div>

          {/* Available models */}
          {models.length > 0 && (
            <div className={styles.card}>
              <div className={styles.cardTitle}>available ollama models</div>
              <div className={styles.modelGrid}>
                {models.map(m => (
                  <div key={m} className={styles.modelTag}>{m}</div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      <div className={styles.actions}>
        <button className={styles.btn} onClick={handleRecheck} disabled={checking}>
          {checking ? <><span className="spinner" /> checking...</> : '⟳ RECHECK'}
        </button>
        <button className={styles.btnAccent} onClick={handleIngest} disabled={ingesting}>
          {ingesting ? <><span className="spinner" /> ingesting...</> : '↺ RE-INGEST PLAYBOOKS'}
        </button>
      </div>

      {ingestMsg && (
        <div className={styles.ingestMsg}
          style={{ color: ingestMsg.startsWith('Failed') ? 'var(--red)' : 'var(--accent)' }}>
          {ingestMsg}
        </div>
      )}
    </div>
  )
}

function Row({ label, children }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowKey}>{label}</span>
      <span className={styles.rowVal}>{children}</span>
    </div>
  )
}

function Val({ val, good }) {
  const color = val === good ? 'var(--accent)' : 'var(--red)'
  return <span style={{ color }}>{val || '—'}</span>
}

function Bool({ val }) {
  return val === true
    ? <span style={{ color: 'var(--accent)' }}>yes</span>
    : val === false
    ? <span style={{ color: 'var(--red)' }}>no</span>
    : <span style={{ color: 'var(--text3)' }}>—</span>
}