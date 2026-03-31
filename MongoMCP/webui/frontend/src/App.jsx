import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import JsonDataRenderer from './JsonDataRenderer'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState(null)
  const [answer, setAnswer] = useState(null)
  const [mapData, setMapData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [streamedOutput, setStreamedOutput] = useState('')
  const [status, setStatus] = useState(null)
  const [liveMessage, setLiveMessage] = useState('')
  const [patternSaved, setPatternSaved] = useState(false)
  const [patternSaving, setPatternSaving] = useState(false)

  function parseMaybeJson(value) {
    if (typeof value !== 'string') return value
    const trimmed = value.trim()
    if (!trimmed) return null
    try {
      return JSON.parse(trimmed)
    } catch {
      return value
    }
  }

  function consumeStreamPayload(rawPayload) {
    let obj = parseMaybeJson(rawPayload)
    obj = parseMaybeJson(obj)
    if (!obj || typeof obj !== 'object') return

    if (obj.history !== undefined && obj.history !== null) {
      setHistory(obj.history)
    }

    if (obj.status !== undefined) setStatus(obj.status)
    if (obj.message !== undefined) {
      const msg = String(obj.message)
      setLiveMessage(msg)
      setStreamedOutput((prev) => (prev ? prev + '\n' : '') + msg)
    }

    const content = parseMaybeJson(obj.content)
    if (content && typeof content === 'object') {
      setAnswer(content.text || null)
      const jd = content.jsondata ?? content.jsonData
      if (jd && (jd.jsonDataType || jd.jsondataType)) {
        setMapData(jd)
      }
    }
  }

  async function savePattern() {
    try {
      setPatternSaving(true)
      const res = await fetch(`${API_URL}/pattern/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to save pattern')
      setPatternSaved(true)
    } catch (e) {
      setError(String(e))
    } finally {
      setPatternSaving(false)
    }
  }

  async function submitQuestion() {
    setLoading(true)
    setError(null)
    setStreamedOutput('')
    setAnswer(null)
    setMapData(null)
    setStatus(null)
    setLiveMessage('')
    setPatternSaved(false)
    try {
      // Try streaming endpoint first
      const res = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: question, history }),
      })
      
      if (!res.ok && res.status !== 404) {
        throw new Error(data.error || 'API error')
      }
      
      // If streaming available, read NDJSON stream and update live/status/history
      if (res.ok && res.body) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          buf += chunk

          const lines = buf.split('\n')
          buf = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed) continue
            const payload = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed
            consumeStreamPayload(payload)
          }
        }

        // Handle any remaining partial buffer at stream end.
        if (buf && buf.trim()) {
          const trimmed = buf.trim()
          const payload = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed
          consumeStreamPayload(payload)
        }
      } else {
        // Fall back to regular endpoint
        const res2 = await fetch(`${API_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: question, history }),
        })
        const data = await res2.json()
        if (!res2.ok) throw new Error(data.error || 'API error')
        setAnswer(data.answer || data.message || null)
        setMapData(null)
        setHistory(data.history || history)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function resetHistory() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/history/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to reset history')

      setHistory([])
      setStatus(null)
      setLiveMessage('')
      setStreamedOutput('')
      setAnswer(null)
      setMapData(null)
      setPatternSaved(false)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ fontFamily: 'Arial' }}>
      {/* Header with logo */}
      <header style={{
        backgroundColor: '#fff',
        padding: '16px 24px',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        marginBottom: '24px',
        display: 'flex',
        alignItems: 'center',
        gap: '16px'
      }}>
        <img
          src="/leaflogo.png"
          alt="Logo"
          style={{ height: '48px', width: 'auto' }}
        />
        <h1 style={{ margin: 0, fontSize: '24px', fontWeight: '600' }}>
          MCP Query and Viewer
        </h1>
      </header>

      <div style={{ padding: '0 24px' }}>
      <div>
        <textarea
          placeholder="Enter question or command (clear, cache stats, cache clear)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submitQuestion();
            }
          }}
          rows={4}
          style={{ width: '100%' }}
        />
      </div>
      <div style={{ marginTop: 8 }}>
        <button onClick={submitQuestion} disabled={loading}>
          {loading ? '⏳ Processing...' : 'Send'}
        </button>
        <button onClick={resetHistory} disabled={loading} style={{ marginLeft: 8 }}>
          Reset History
        </button>
      </div>

      {error && <div style={{ color: 'red', marginTop: 8 }}>❌ {error}</div>}

      {(streamedOutput || status) && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ minWidth: 180, backgroundColor: '#fff', padding: 10, borderRadius: 6, border: '1px solid #eee' }}>
              <strong>Status</strong>
              <div style={{ marginTop: 6, fontSize: 14 }}>{status ?? '—'}</div>
            </div>

            <div style={{ flex: 1, backgroundColor: '#f0f0f0', padding: 8, borderRadius: 4 }}>
              <strong>Live Output:</strong>
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>{streamedOutput}</pre>
            </div>
          </div>
        </div>
      )}

      {mapData && (
        <div style={{ marginTop: 12 }}>
          <div style={{ marginTop: 8 }}>
            <JsonDataRenderer jsonData={mapData} />
          </div>
        </div>
      )}

      {answer && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <strong>Answer:</strong>
            <button
              onClick={savePattern}
              disabled={patternSaved || patternSaving || loading}
              title={patternSaved ? 'Pattern saved' : patternSaving ? 'Saving...' : 'Save this pattern for future queries'}
              style={{
                background: 'none',
                border: 'none',
                cursor: (patternSaved || patternSaving) ? 'default' : 'pointer',
                fontSize: 22,
                padding: '2px 6px',
                opacity: patternSaving ? 0.3 : 1,
                filter: 'none',
                color: 'inherit',
                transition: 'opacity 0.2s ease',
              }}
            >
              {patternSaved ? '✅ Saved' : patternSaving ? '⏳' : '👍'}
            </button>
          </div>
          <div
            className="markdown-content"
            style={{
              marginTop: 8,
              backgroundColor: '#fff',
              padding: 12,
              borderRadius: 6,
              border: '1px solid #eee',
              lineHeight: '1.6'
            }}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        <strong>History (JSON):</strong>
        <div style={{ marginTop: 8, maxHeight: 300, overflow: 'auto', backgroundColor: '#f9f9f9', padding: 8, borderRadius: 4 }}>
          {history ? (
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(history, null, 2)}</pre>
          ) : (
            <div style={{ color: '#666' }}>No history yet</div>
          )}
        </div>
      </div>
      </div>
    </div>
  )
}
