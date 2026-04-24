import styles from './Topbar.module.css'

export default function Topbar({ apiStatus, pbCount, model }) {
  const statusColor = apiStatus === 'online'
    ? 'var(--accent)'
    : apiStatus === 'offline'
    ? 'var(--red)'
    : 'var(--amber)'

  return (
    <header className={styles.topbar}>
      <div className={styles.dot} style={{ background: statusColor, boxShadow: `0 0 6px ${statusColor}` }} />
      <div className={styles.logo}>RAG<span>SOC</span></div>
      <span className={styles.sub}>// threat intelligence assistant</span>

      <div className={styles.meta}>
        <MetaItem label="API">
          <span style={{ color: statusColor }}>{apiStatus}</span>
        </MetaItem>
        <MetaItem label="MODEL">{model || 'llama3'}</MetaItem>
        <MetaItem label="KB">{pbCount ? `${pbCount} playbooks` : '—'}</MetaItem>
        <MetaItem label="TOP-K">2</MetaItem>
      </div>
    </header>
  )
}

function MetaItem({ label, children }) {
  return (
    <div className={styles.metaItem}>
      <span>{label}</span>
      <span className={styles.metaVal}>{children}</span>
    </div>
  )
}
