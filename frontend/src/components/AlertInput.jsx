import { useState, useEffect } from 'react'
import styles from './AlertInput.module.css'

const DEFAULT_MODEL = 'llama3:latest'

export default function AlertInput({ onAnalyze, prefill, loading, availableModels = [] }) {
  const [text,  setText]  = useState('')
  const [model, setModel] = useState(DEFAULT_MODEL)

  useEffect(() => {
    if (prefill) setText(prefill)
  }, [prefill])

  // Keep selected model valid when model list loads
  useEffect(() => {
    if (availableModels.length && !availableModels.includes(model)) {
      const fallback = availableModels.find(m => m.startsWith('llama3')) || availableModels[0]
      setModel(fallback)
    }
  }, [availableModels, model])

  function handleKeyDown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      if (text.trim() && !loading) onAnalyze(text.trim(), model)
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.arrow}>&gt;</span> alert log input
        <span className={styles.hint}>ctrl+enter to run</span>
      </div>

      <div className={styles.body}>
        <textarea
          className={styles.input}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={PLACEHOLDER}
          spellCheck={false}
        />
      </div>

      <div className={styles.footer}>
        <button
          className={styles.analyzeBtn}
          onClick={() => onAnalyze(text.trim(), model)}
          disabled={!text.trim() || loading}
        >
          {loading ? <><span className="spinner" /> analyzing...</> : '▶ ANALYZE'}
        </button>
        <button
          className={styles.clearBtn}
          onClick={() => setText('')}
          disabled={loading}
        >
          CLEAR
        </button>

        {/* Model selector */}
        {availableModels.length > 0 && (
          <select
            className={styles.modelSelect}
            value={model}
            onChange={e => setModel(e.target.value)}
            disabled={loading}
          >
            {availableModels.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}

        <span className={styles.charCount}>{text.length} chars</span>
      </div>
    </div>
  )
}

const PLACEHOLDER = `Paste raw alert log, JSON event, or describe the incident...

Example:
{
  "timestamp": "2024-01-15T03:22:11Z",
  "event_type": "authentication_failure",
  "source_ip": "185.220.101.45",
  "attempts": 847,
  "target": "ssh/22",
  "user": "root"
}`