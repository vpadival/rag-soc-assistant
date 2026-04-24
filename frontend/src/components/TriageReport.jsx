import styles from './TriageReport.module.css'

const DISTANCE_THRESHOLD = 0.5  // playbooks above this are weak matches

function detectSeverity(text) {
  const t = (text || '').toUpperCase()
  if (t.includes('CRITICAL')) return 'CRITICAL'
  if (t.includes(' HIGH') || t.includes('HIGH ')) return 'HIGH'
  if (t.includes('MEDIUM')) return 'MEDIUM'
  if (t.includes(' LOW') || t.includes('LOW ')) return 'LOW'
  return 'UNKNOWN'
}

function sevColor(sev) {
  return {
    CRITICAL: 'var(--red)',
    HIGH:     'var(--amber)',
    MEDIUM:   'var(--blue)',
    LOW:      'var(--accent)',
  }[sev] || 'var(--text3)'
}

function ThreatBar({ sev }) {
  const levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
  const idx    = levels.indexOf(sev)
  const cls    = { CRITICAL: styles.segCrit, HIGH: styles.segHigh, MEDIUM: styles.segMed, LOW: styles.segLow }[sev]
  return (
    <div className={styles.threatBar}>
      {levels.map((_, i) => (
        <div key={i} className={`${styles.seg} ${i <= idx && idx >= 0 ? cls : ''}`} />
      ))}
    </div>
  )
}

function SkeletonLoader() {
  return (
    <div className={styles.skeleton}>
      {[80, 100, 60, 90, 70, 50, 85].map((w, i) => (
        <div key={i} className={styles.skLine} style={{ width: `${w}%` }} />
      ))}
    </div>
  )
}

function EmptyState() {
  return (
    <div className={styles.empty}>
      <div className={styles.emptyIcon}>◆</div>
      <div className={styles.emptyText}>
        paste an alert log on the left<br />
        and hit ANALYZE to generate<br />
        a structured triage report
      </div>
    </div>
  )
}

function SourceTag({ source, weak }) {
  const label = typeof source === 'object'
    ? (source.id || source.title || JSON.stringify(source))
    : String(source)
  return (
    <span className={`${styles.sourceTag} ${weak ? styles.sourceTagWeak : ''}`} title={weak ? 'Weak match (distance > 0.5)' : ''}>
      {label}{weak ? ' ·' : ''}
    </span>
  )
}

function PlaybookCard({ pb }) {
  const sev      = (pb.severity || '').toUpperCase()
  const badgeCls = sev === 'CRITICAL' ? 'badge--crit' : sev === 'HIGH' ? 'badge--high' : sev === 'MEDIUM' ? 'badge--med' : 'badge--low'
  const weak     = pb.distance != null && pb.distance > DISTANCE_THRESHOLD
  const distPct  = pb.distance != null ? Math.min(100, Math.round(pb.distance * 100)) : null

  return (
    <div className={`${styles.pbCard} ${weak ? styles.pbCardWeak : ''}`}>
      <div className={styles.pbCardHeader}>
        <span className={styles.pbCardTitle}>{pb.title || pb.id}</span>
        {sev && <span className={`badge ${badgeCls}`}>{sev}</span>}
        {pb.mitre_technique && <span className={styles.mitre}>{pb.mitre_technique}</span>}
        {distPct != null && (
          <span className={`${styles.distBadge} ${weak ? styles.distWeak : styles.distStrong}`}>
            {weak ? 'weak' : 'strong'} · {distPct}%
          </span>
        )}
      </div>
      {weak && (
        <div className={styles.weakNote}>
          Low similarity match — may not be directly relevant to this alert
        </div>
      )}
    </div>
  )
}

export default function TriageReport({ result, loading, error, status }) {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.arrow}>&gt;</span> triage report
        {status && <span className={styles.status}>{status}</span>}
      </div>
      <div className={styles.body}>
        {loading && <SkeletonLoader />}
        {error && !loading && (
          <div className={styles.errorBox}>
            <strong>Analysis failed</strong><br /><br />
            {error}
          </div>
        )}
        {!loading && !error && !result && <EmptyState />}
        {!loading && !error && result && <ReportContent result={result} />}
      </div>
    </div>
  )
}

function ReportContent({ result }) {
  const model      = result.model_used || result.model || 'llama3'
  const sources    = result.retrieved_playbooks || result.sources || []
  const analysis   = result.analysis || {}
  const mitigation = analysis.mitigation || []
  const sev        = (analysis.severity || detectSeverity(JSON.stringify(result))).toUpperCase()
  const color      = sevColor(sev)

  const strongSources = sources.filter(s => s.distance == null || s.distance <= DISTANCE_THRESHOLD)
  const weakSources   = sources.filter(s => s.distance != null && s.distance > DISTANCE_THRESHOLD)

  return (
    <div className={styles.reportStack}>

      {/* Triage summary */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle} style={{ color }}>◆ triage summary</span>
          <div className={styles.threatMeta}>
            <ThreatBar sev={sev} />
            <span style={{ fontSize: 9, color, fontWeight: 600 }}>{sev}</span>
          </div>
        </div>
        <div className={styles.cardBody}>
          <Row label="model">{model}</Row>
          <Row label="attack type">{analysis.attack_type || '—'}</Row>
          <Row label="mitre">{analysis.mitre_attack || '—'}</Row>
          <Row label="detection">{analysis.detection_recommendation || '—'}</Row>
          <Row label="retrieved">
            <div className={styles.sourceRow}>
              {sources.length === 0 && <span className={styles.dim}>none retrieved</span>}
              {sources.map((s, i) => (
                <SourceTag
                  key={i}
                  source={s}
                  weak={s.distance != null && s.distance > DISTANCE_THRESHOLD}
                />
              ))}
            </div>
          </Row>
        </div>
      </div>

      {/* Retrieval quality notice */}
      {weakSources.length > 0 && strongSources.length === 0 && (
        <div className={styles.qualityNote}>
          ⚠ Both retrieved playbooks are weak matches. Consider rephrasing your alert
          with more specific threat indicators, or expanding the playbook knowledge base.
        </div>
      )}

      {/* Explanation */}
      {analysis.explanation && (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle} style={{ color: 'var(--text2)' }}>explanation</span>
          </div>
          <div className={`${styles.cardBody} ${styles.rawResponse}`}>{analysis.explanation}</div>
        </div>
      )}

      {/* Mitigation steps */}
      {mitigation.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle} style={{ color: 'var(--accent)' }}>mitigation steps</span>
          </div>
          <div className={styles.cardBody}>
            <ol className={styles.stepList}>
              {mitigation.map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </div>
        </div>
      )}

      {/* Matched playbooks */}
      {sources.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.cardTitle} style={{ color: 'var(--blue)' }}>matched playbooks</span>
            <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text3)' }}>
              threshold &lt;0.5 = strong
            </span>
          </div>
          <div className={styles.cardBody}>
            {sources.map((s, i) => (
              <PlaybookCard key={i} pb={typeof s === 'object' ? s : { title: String(s) }} />
            ))}
          </div>
        </div>
      )}

    </div>
  )
}

function Row({ label, children }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowKey}>{label}</span>
      <div className={styles.rowVal}>{children}</div>
    </div>
  )
}