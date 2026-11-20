import React, { useState } from 'react'

export default function ChangePassword({ apiUrl, username, onChanged, onLogout }) {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    if (busy) return
    if (next !== confirm) { setError('New passwords do not match'); return }
    setBusy(true)
    setError(null)
    try {
      const res = await fetch(`${apiUrl}/auth/change-password`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_password: current, new_password: next }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      onChanged()
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setBusy(false)
    }
  }

  const inputStyle = {
    width: '100%', boxSizing: 'border-box', fontSize: 14, padding: '10px 12px',
    border: '1.5px solid var(--mdb-border)', borderRadius: 8,
    background: 'var(--mdb-surface-2)', color: 'var(--mdb-text)', outline: 'none',
    margin: '4px 0 14px',
  }
  const labelStyle = { fontSize: 13, color: 'var(--mdb-muted)' }

  return (
    <div style={{ minHeight: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <form onSubmit={submit} style={{
        width: 340, background: 'var(--mdb-surface)', border: '1px solid var(--mdb-border)',
        borderRadius: 12, padding: 28, boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
      }}>
        <h1 style={{ margin: '0 0 6px', fontSize: 18, fontWeight: 700 }}>Change your password</h1>
        <p style={{ margin: '0 0 18px', fontSize: 13, color: 'var(--mdb-muted)' }}>
          {username}, you must choose a new password before continuing (minimum 8 characters).
        </p>
        <label style={labelStyle}>Current password</label>
        <input type="password" style={inputStyle} value={current} onChange={e => setCurrent(e.target.value)} autoComplete="current-password" autoFocus />
        <label style={labelStyle}>New password</label>
        <input type="password" style={inputStyle} value={next} onChange={e => setNext(e.target.value)} autoComplete="new-password" />
        <label style={labelStyle}>Confirm new password</label>
        <input type="password" style={inputStyle} value={confirm} onChange={e => setConfirm(e.target.value)} autoComplete="new-password" />
        {error && (
          <div style={{ color: '#ff9b8f', background: '#3a1815', border: '1px solid #6b2b25', borderRadius: 8, padding: '8px 12px', marginBottom: 14, fontSize: 13 }}>
            {error}
          </div>
        )}
        <button className="btn-send" type="submit" disabled={busy || !current || !next || !confirm} style={{ width: '100%' }}>
          {busy ? 'Saving…' : 'Change password'}
        </button>
        <button type="button" className="btn-secondary" onClick={onLogout} style={{ width: '100%', marginTop: 8 }}>
          Log out
        </button>
      </form>
    </div>
  )
}
