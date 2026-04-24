import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import Topbar       from './components/Topbar'
import Sidebar      from './components/Sidebar'
import AlertInput   from './components/AlertInput'
import TriageReport from './components/TriageReport'
import HealthPanel  from './components/HealthPanel'
import styles from './App.module.css'

const HISTORY_KEY = 'soc-history'

function useAppState() {
  const [health,    setHealth]    = useState(null)
  const [playbooks, setPlaybooks] = useState([])
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    api.health()
      .then(h  => { setHealth(h); setApiStatus('online') })
      .catch(() => setApiStatus('offline'))

    api.playbooks()
      .then(d => setPlaybooks(d.playbooks || d || []))
      .catch(() => {})
  }, [])

  const triggerIngest = useCallback(async () => {
    const d = await api.ingest()
    api.playbooks().then(d2 => setPlaybooks(d2.playbooks || d2 || [])).catch(() => {})
    return d
  }, [])

  const recheckHealth = useCallback(() =>
    api.health()
      .then(h  => { setHealth(h); setApiStatus('online') })
      .catch(() => { setHealth(null); setApiStatus('offline') })
  , [])

  return { health, playbooks, apiStatus, triggerIngest, recheckHealth }
}

export default function App() {
  const { health, playbooks, apiStatus, triggerIngest, recheckHealth } = useAppState()

  const [tab,      setTab]      = useState('triage')
  const [activePb, setActivePb] = useState(null)
  const [prefill,  setPrefill]  = useState('')
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [status,   setStatus]   = useState('')

  // Persistent history
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') }
    catch { return [] }
  })

  function saveHistory(updated) {
    setHistory(updated)
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(updated)) } catch {}
  }

  // Pull available models from health response
  const availableModels = health?.ollama?.available_models || []
  // Topbar shows active model from result, or first available
  const activeModel = result?.model_used || availableModels[0] || 'llama3'

  const handleAnalyze = useCallback(async (text, model) => {
    if (!text) return
    setLoading(true); setError(null); setResult(null); setStatus('analyzing...')
    try {
      const d = await api.analyze(text, model)
      setResult(d)
      const ts = new Date().toLocaleTimeString()
      setStatus('done · ' + ts)
      saveHistory(
        [{ ts, q: text, result: d }, ...history].slice(0, 10)
      )
    } catch (e) {
      setError(
        e.message.includes('fetch')
          ? 'Cannot reach API at http://localhost:8000.\n\nStart it with:\npython -m uvicorn api:app --reload --host 0.0.0.0 --port 8000'
          : e.message
      )
      setStatus('error')
    }
    setLoading(false)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history])

  function handleSelectPlaybook(i) {
    setActivePb(i)
    const pb = playbooks[i]
    if (pb) setPrefill(
      `Analyze a ${pb.title} incident.\n\nThreat type: ${pb.title}\nSeverity: ${pb.severity}\nMITRE: ${pb.mitre_technique || 'N/A'}`
    )
  }

  function handleSelectHistory(i) {
    const h = history[i]
    setPrefill(h.q); setResult(h.result); setError(null); setStatus('loaded from history')
  }

  return (
    <div className={styles.shell}>
      <Topbar apiStatus={apiStatus} pbCount={playbooks.length} model={activeModel} />

      <div className={styles.main}>
        <Sidebar
          playbooks={playbooks}
          history={history}
          activeIdx={activePb}
          onSelectPlaybook={handleSelectPlaybook}
          onSelectHistory={handleSelectHistory}
        />

        <div className={styles.content}>
          <div className={styles.tabBar}>
            <span className={styles.contentTitle}>// alert triage console</span>
            <div className={styles.tabs}>
              {['triage', 'health'].map(t => (
                <button
                  key={t}
                  className={`${styles.tab} ${tab === t ? styles.tabActive : ''}`}
                  onClick={() => setTab(t)}
                >
                  {t === 'triage' ? 'paste log' : 'api health'}
                </button>
              ))}
            </div>
          </div>

          {tab === 'triage' && (
            <div className={styles.workspace}>
              <AlertInput
                onAnalyze={handleAnalyze}
                prefill={prefill}
                loading={loading}
                availableModels={availableModels}
              />
              <TriageReport result={result} loading={loading} error={error} status={status} />
            </div>
          )}

          {tab === 'health' && (
            <HealthPanel
              health={health}
              apiStatus={apiStatus}
              onRecheck={recheckHealth}
              onIngest={triggerIngest}
            />
          )}
        </div>
      </div>
    </div>
  )
}