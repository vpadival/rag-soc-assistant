import styles from './Sidebar.module.css'

const SEV_ORDER = [
  { key: 'CRITICAL', color: 'var(--red)' },
  { key: 'HIGH',     color: 'var(--amber)' },
  { key: 'MEDIUM',   color: 'var(--blue)' },
]

function sevBadgeClass(sev) {
  const s = (sev || '').toUpperCase()
  if (s === 'CRITICAL') return 'badge--crit'
  if (s === 'HIGH')     return 'badge--high'
  if (s === 'MEDIUM')   return 'badge--med'
  return 'badge--low'
}

export default function Sidebar({ playbooks, history, activeIdx, onSelectPlaybook, onSelectHistory }) {
  const counts = SEV_ORDER.reduce((acc, { key }) => {
    acc[key] = playbooks.filter(p => (p.severity || '').toUpperCase() === key).length
    return acc
  }, {})
  const maxCount = Math.max(...Object.values(counts), 1)

  return (
    <aside className={styles.sidebar}>
      {/* Severity index */}
      <div className={styles.section}>
        <div className={styles.label}>severity index</div>
        {SEV_ORDER.map(({ key, color }) => (
          <div className={styles.sevRow} key={key}>
            <span className={styles.sevName}>{key}</span>
            <div className={styles.sevBarWrap}>
              <div
                className={styles.sevBar}
                style={{ width: `${(counts[key] / maxCount) * 100}%`, background: color }}
              />
            </div>
            <span className={styles.sevCount} style={{ color }}>{counts[key]}</span>
          </div>
        ))}
      </div>

      {/* Playbook corpus */}
      <div className={styles.section}>
        <div className={styles.label}>playbook corpus</div>
      </div>
      <div className={styles.pbList}>
        {playbooks.length === 0 && (
          <div className={styles.empty}>loading playbooks...</div>
        )}
        {playbooks.map((pb, i) => (
          <div
            key={pb.id || i}
            className={`${styles.pbItem} ${activeIdx === i ? styles.pbActive : ''}`}
            onClick={() => onSelectPlaybook(i)}
          >
            <div className={styles.pbTitle}>{pb.title || pb.id}</div>
            <div className={styles.pbMeta}>
              <span className={`badge ${sevBadgeClass(pb.severity)}`}>
                {(pb.severity || '').toUpperCase()}
              </span>
              <span className="mitre">{pb.mitre_technique || ''}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Recent queries */}
      <div className={styles.historySection}>
        <div className={styles.label}>recent queries</div>
        {history.length === 0 && <div className={styles.empty}>no queries yet</div>}
        {history.slice(0, 6).map((h, i) => (
          <div key={i} className={styles.historyItem} onClick={() => onSelectHistory(i)}>
            <div className={styles.historyTs}>{h.ts}</div>
            <div className={styles.historyQ}>{h.q.substring(0, 55)}{h.q.length > 55 ? '…' : ''}</div>
          </div>
        ))}
      </div>
    </aside>
  )
}
