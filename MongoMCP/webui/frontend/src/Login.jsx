import React, { useState } from 'react'

export default function Login({ apiUrl, onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      const res = await fetch(`${apiUrl}/auth/login`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
      onLogin(data.username, data.must_change_password)
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
  }

  return (
    <div style={{ minHeight: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <form onSubmit={submit} style={{
        width: 340, background: 'var(--mdb-surface)', border: '1px solid var(--mdb-border)',
        borderRadius: 12, padding: 28, boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
      }}>
        <div style={{ textAlign: 'center', marginBottom: 20 }}>
          <img src="/leaflogo.png" alt="Logo" style={{ height: 56, width: 'auto' }} />
          <h1 style={{ margin: '10px 0 0', fontSize: 18, fontWeight: 700 }}>MongoDB Atlas MCP AI Demo</h1>
        </div>
        <label style={{ fontSize: 13, color: 'var(--mdb-muted)' }}>Username</label>
        <input
          style={{ ...inputStyle, margin: '4px 0 14px' }}
          value={username}
          onChange={e => setUsername(e.target.value)}
          autoComplete="username"
          autoFocus
        />
        <label style={{ fontSize: 13, color: 'var(--mdb-muted)' }}>Password</label>
        <input
          type="password"
          style={{ ...inputStyle, margin: '4px 0 18px' }}
          value={password}
          onChange={e => setPassword(e.target.value)}
          autoComplete="current-password"
        />
        {error && (
          <div style={{ color: '#ff9b8f', background: '#3a1815', border: '1px solid #6b2b25', borderRadius: 8, padding: '8px 12px', marginBottom: 14, fontSize: 13 }}>
            {error}
          </div>
        )}
        <button className="btn-send" type="submit" disabled={busy || !username || !password} style={{ width: '100%' }}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </div>
  )
}
