import React, { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState(null)
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [streamedOutput, setStreamedOutput] = useState('')
  const [status, setStatus] = useState(null)
  const [liveMessage, setLiveMessage] = useState('')

  async function submitQuestion() {
    setLoading(true)
    setError(null)
    setStreamedOutput('')
    setAnswer(null)
    setStatus(null)
    setLiveMessage('')
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

          // process complete lines (NDJSON)
          const lines = buf.split('\n')
          buf = lines.pop() // leftover

          for (const line of lines) {
            if (!line || !line.trim()) continue
            try {
              const obj = JSON.parse(line)

              // If response carries history, replace the history viewer
              if (obj.history !== undefined) {
                setHistory(obj.history)
              }

              if (obj.status !== undefined) setStatus(obj.status)
              if (obj.message !== undefined) setLiveMessage(String(obj.message))

              // Only show status/message/answer in the live output
              const liveParts = []
              //if (obj.status !== undefined) liveParts.push(String(obj.status))
              if (obj.message !== undefined) liveParts.push(String(obj.message))
              if (obj.answer !== undefined) {
                liveParts.push(String(obj.answer))
                setAnswer(obj.answer)
              }

              if (liveParts.length) {
                setStreamedOutput((prev) => (prev ? prev + '\n' : '') + liveParts.join(' '))
              }
            } catch (e) {
              // Non-JSON line — append raw
              setStreamedOutput((prev) => prev + line)
            }
          }
        }

        // handle any remaining buffer as final JSON or raw
        if (buf && buf.trim()) {
          try {
            const data = JSON.parse(buf)
            if (data.history !== undefined) setHistory(data.history)
            if (data.answer !== undefined) setAnswer(data.answer)
            if (data.status !== undefined) setStatus(data.status)
            //if (data.message !== undefined) setLiveMessage(String(data.message))
            const liveParts = []
            if (data.message !== undefined) liveParts.push(String(data.message))
            if (liveParts.length) setStreamedOutput((prev) => (prev ? prev + '\n' : '') + liveParts.join(' '))
          } catch {
            setStreamedOutput((prev) => prev + buf)
            setAnswer(buf)
          }
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
        setHistory(data.history || history)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 20, fontFamily: 'Arial' }}>
      <h2>MCP Web UI</h2>
      <div>
        <textarea
          placeholder="Enter question or command (clear, cache stats, cache clear)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={4}
          style={{ width: '100%' }}
        />
      </div>
      <div style={{ marginTop: 8 }}>
        <button onClick={submitQuestion} disabled={loading}>
          {loading ? '⏳ Processing...' : 'Send'}
        </button>
      </div>

      {error && <div style={{ color: 'red', marginTop: 8 }}>❌ {error}</div>}

      {loading && (streamedOutput || status) && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ minWidth: 180, backgroundColor: '#fff', padding: 10, borderRadius: 6, border: '1px solid #eee' }}>
              <strong>Status</strong>
              <div style={{ marginTop: 6, fontSize: 14 }}>{status ?? '—'}</div>
              {liveMessage ? (
                <div style={{ marginTop: 8, color: '#333', whiteSpace: 'pre-wrap' }}>{liveMessage}</div>
              ) : null}
            </div>

            <div style={{ flex: 1, backgroundColor: '#f0f0f0', padding: 8, borderRadius: 4 }}>
              <strong>Live Output:</strong>
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>{streamedOutput}</pre>
            </div>
          </div>
        </div>
      )}

      {answer && (
        <div style={{ marginTop: 12 }}>
          <strong>Answer:</strong>
          <div
            style={{ marginTop: 8, backgroundColor: '#fff', padding: 12, borderRadius: 6, border: '1px solid #eee' }}
            dangerouslySetInnerHTML={{ __html: answer }}
          />
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
  )
}
