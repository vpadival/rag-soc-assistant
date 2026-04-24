const API_BASE = 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export const api = {
  health:    ()            => request('/health'),
  playbooks: ()            => request('/playbooks'),
  ingest:    ()            => request('/ingest', { method: 'POST' }),
  analyze:   (query, model) =>
    request('/analyze', {
      method: 'POST',
      body: JSON.stringify(model ? { alert: query, model } : { alert: query }),
    }),
}