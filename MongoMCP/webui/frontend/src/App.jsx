import React, { useState, useRef, useEffect } from 'react'
import ChatMessage from './ChatMessage'

const API_URL = import.meta.env.VITE_API_URL || ''

function getCookie(name) {
  const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'))
  return match ? decodeURIComponent(match[1]) : null
}

function setCookie(name, value, days = 365) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString()
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`
}

const SESSION_GREETING = (uname) => `Hi, my username is "${uname}".`

/** Strip [JSON_DATA_START]...[JSON_DATA_END] blocks from assistant text. */
function stripJsonDataBlock(text) {
  return text.replace(/\[JSON_DATA_START\][\s\S]*?\[JSON_DATA_END\]/g, '').trim()
}

/**
 * Parse a Bedrock history array into structured conversation turns.
 * Each turn: { userText, assistantText, toolCalls: [{name, input, result}] }
 *
 * Bedrock history groups:
 *   user(text) → assistant(text + toolUse*) → user(toolResult*) → assistant(text) → ...
 * We collapse all assistant + toolResult exchanges for a given user question into one turn.
 */
function parseHistoryToTurns(history) {
  if (!Array.isArray(history) || history.length === 0) return []

  const turns = []
  let i = 0

  while (i < history.length) {
    const msg = history[i]
    if (!msg || msg.role !== 'user') { i++; continue }

    const userContent = Array.isArray(msg.content) ? msg.content : []
    const isToolResultOnly = userContent.length > 0 && userContent.every(
      b => typeof b === 'object' && b !== null && 'toolResult' in b
    )
    if (isToolResultOnly) { i++; continue }

    const userText = userContent
      .filter(b => typeof b === 'string' || (typeof b === 'object' && b !== null && 'text' in b))
      .map(b => (typeof b === 'string' ? b : b.text || ''))
      .join('\n')
      .trim()

    if (!userText) { i++; continue }

    const startIndex = i
    const toolCallsMap = {}
    let assistantText = ''
    i++

    while (i < history.length) {
      const next = history[i]
      if (!next) { i++; continue }

      if (next.role === 'assistant') {
        const blocks = Array.isArray(next.content) ? next.content : []
        for (const block of blocks) {
          if (!block || typeof block !== 'object') continue
          if ('text' in block) {
            assistantText = (assistantText ? assistantText + '\n' : '') + block.text
          } else if (block.type === 'toolUse' || 'toolUse' in block) {
            const tu = block.toolUse || block
            const _tid = tu.toolUseId || tu.id || String(Object.keys(toolCallsMap).length)
            toolCallsMap[_tid] = {
              toolUseId: _tid,
              name: tu.name,
              input: tu.input,
              result: undefined,
            }
          }
        }
        i++
      } else if (next.role === 'user') {
        const blocks = Array.isArray(next.content) ? next.content : []
        const isResultOnly = blocks.length > 0 && blocks.every(
          b => typeof b === 'object' && b !== null && 'toolResult' in b
        )
        if (!isResultOnly) break

        for (const block of blocks) {
          const tr = block.toolResult || block
          const tid = tr.toolUseId
          const resultContent = Array.isArray(tr.content)
            ? tr.content.map(c => {
                if (typeof c === 'string') return c
                if (typeof c === 'object' && c !== null && 'text' in c) return c.text
                return JSON.stringify(c)
              }).join('\n')
            : (typeof tr.content === 'string' ? tr.content : JSON.stringify(tr.content))
          // Unwrap up to 3 levels of JSON string encoding
          let parsed = resultContent
          for (let j = 0; j < 3; j++) {
            if (typeof parsed !== 'string') break
            try { parsed = JSON.parse(parsed) } catch { break }
          }
          if (tid && toolCallsMap[tid]) toolCallsMap[tid].result = parsed
        }
        i++
      } else {
        break
      }
    }

    turns.push({
      startIndex,
      userText,
      assistantText: assistantText.trim() ? stripJsonDataBlock(assistantText.trim()) : null,
      toolCalls: Object.values(toolCallsMap),
    })
  }

  return turns
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [mapData, setMapData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [streamedOutput, setStreamedOutput] = useState('')
  const [status, setStatus] = useState(null)
  const [liveMessage, setLiveMessage] = useState('')
  const [patternSaved, setPatternSaved] = useState(false)
  const [patternSaving, setPatternSaving] = useState(false)
  const [userId] = useState(() => {
    let id = localStorage.getItem('mcp_user_id')
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('mcp_user_id', id) }
    return id
  })
  const [sessionId, setSessionId] = useState(() => {
    // Persist so a page refresh keeps the same conversation thread (the backend
    // reloads the deterministic checkpoint by session_id). 'Clear chat' mints a new one.
    let id = localStorage.getItem('mcp_session_id')
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('mcp_session_id', id) }
    return id
  })
  const [username, setUsername] = useState(() => getCookie('mcp_username') || 'demo-user')
  const [usernameDraft, setUsernameDraft] = useState(() => getCookie('mcp_username') || 'demo-user')
  // When an auth provider is enabled the backend supplies a verified identity; the
  // username field is then locked and must not be edited client-side.
  const [authLocked, setAuthLocked] = useState(false)
  const [feedbackGiven, setFeedbackGiven] = useState(null)
  const [reasoningSteps, setReasoningSteps] = useState([])
  const [pendingQuestion, setPendingQuestion] = useState(null)
  // Which dynamic-function stage the backend surfaces to the model. 'dev' also
  // surfaces in-development fn_ functions so they can be tested before publishing.
  // Persisted so a reload keeps the UI authoritative across gunicorn workers.
  const [functionStage, setFunctionStage] = useState(() => localStorage.getItem('mcp_function_stage') || 'prod')
  const [stageSwitching, setStageSwitching] = useState(false)

  // MCP token & VS Code config modal state
  const [mcpModalOpen, setMcpModalOpen] = useState(false)
  const [mcpConfigData, setMcpConfigData] = useState(null)
  const [mcpLoading, setMcpLoading] = useState(false)
  const [mcpError, setMcpError] = useState(null)
  const [copiedConfig, setCopiedConfig] = useState(false)
  const [copiedToken, setCopiedToken] = useState(false)
  const [regeneratingToken, setRegeneratingToken] = useState(false)

  const modelId = import.meta.env.VITE_LLM_MODEL_ID || ''

  const lastQuestionRef = useRef('')
  const autoGreetingFiredRef = useRef(false)
  const liveLogRef = useRef(null)
  const chatBodyRef = useRef(null)

  const frozenReasoningRef = useRef([])  // steps for the current in-flight turn
  const currentMapDataRef = useRef(null) // in-flight map data (avoids stale effect closure)
  // Captured from the streamed payloads so the committed turn uses the authoritative
  // final answer + history rather than depending on the (trimmed) round-trip.
  const finalHistoryRef = useRef(null)
  const finalAnswerTextRef = useRef('')
  // Per-tool execution times (toolUseId -> ms) for the in-flight turn, delivered
  // in the final content.tool_timings and joined into the committed turn's toolCalls.
  const toolTimingsRef = useRef({})
  const clearHistoryRef = useRef(false)

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight
    }
  }, [transcript, streamedOutput, loading])

  useEffect(() => {
    if (liveLogRef.current) {
      liveLogRef.current.scrollTop = liveLogRef.current.scrollHeight
    }
  }, [streamedOutput])

  // Resolve the verified identity BEFORE the auto-greeting so the greeting (and the
  // username the LLM sees) uses the verified email rather than the default
  // demo-user. Falls back to the self-declared username when not authenticated.
  useEffect(() => {
    if (autoGreetingFiredRef.current) return
    autoGreetingFiredRef.current = true
    setLoading(true) // show the warming-up spinner from first paint, before /me + greeting
    let cancelled = false
    fetch(`${API_URL}/me`)
      .then(r => (r.ok ? r.json() : null))
      .then(me => {
        let effectiveName = username
        if (!cancelled && me && me.authenticated && me.username) {
          setAuthLocked(true)
          setUsername(me.username)
          setUsernameDraft(me.username)
          effectiveName = me.username
        }
        if (!cancelled) submitQuestion(SESSION_GREETING(effectiveName))
      })
      .catch(() => {
        if (!cancelled) submitQuestion(SESSION_GREETING(username))
      })
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Flush the session checkpoint on tab close / refresh so an odd (un-checkpointed
  // by the 2-turn cadence) tail turn isn't lost. sendBeacon survives page unload.
  useEffect(() => {
    const flush = () => {
      const sid = localStorage.getItem('mcp_session_id')
      const hist = finalHistoryRef.current
      if (!sid || !Array.isArray(hist) || hist.length === 0) return
      const uid = localStorage.getItem('mcp_user_id') || ''
      const uname = getCookie('mcp_username') || username
      try {
        navigator.sendBeacon(
          `${API_URL}/checkpoint/finalize`,
          new Blob(
            [JSON.stringify({ user_id: uid, username: uname, session_id: sid, history: hist })],
            { type: 'application/json' },
          ),
        )
      } catch { /* best-effort */ }
    }
    window.addEventListener('pagehide', flush)
    return () => window.removeEventListener('pagehide', flush)
  }, [username])

  function parseMaybeJson(value) {
    if (typeof value !== 'string') return value
    const trimmed = value.trim()
    if (!trimmed) return null
    try { return JSON.parse(trimmed) } catch { return value }
  }

  function consumeStreamPayload(rawPayload) {
    let obj = parseMaybeJson(rawPayload)
    obj = parseMaybeJson(obj)
    if (!obj || typeof obj !== 'object') return

    if (obj.history !== undefined && obj.history !== null) { setHistory(obj.history); finalHistoryRef.current = obj.history }
    if (obj.status !== undefined) setStatus(obj.status)

    if (obj.message !== undefined) {
      const msg = String(obj.message)
      // Capture ALL live messages as reasoning steps (pinned to this turn when history arrives).
      // Skip heartbeat noise and the final "Completed" marker.
      const skipPatterns = ['LLM is still thinking', 'Querying Claude']
      if (!skipPatterns.some(p => msg.includes(p))) {
        setReasoningSteps(prev => {
          const next = [...prev, { status: obj.status || '', message: msg }]
          frozenReasoningRef.current = next
          return next
        })
      }
      if (obj.status !== 'LLM Reasoning') {
        setLiveMessage(msg)
        setStreamedOutput(prev => (prev ? prev + '\n' : '') + msg)
      }
    }

    const content = parseMaybeJson(obj.content)
    if (content && typeof content === 'object') {
      const jd = content.jsondata ?? content.jsonData
      if (jd && typeof jd === 'object') { currentMapDataRef.current = jd; setMapData(jd) }
      // Authoritative final answer for the answer bubble (survives history trimming).
      if (typeof content.text === 'string' && content.text.trim()) finalAnswerTextRef.current = content.text
      if (content.tool_timings && typeof content.tool_timings === 'object') {
        toolTimingsRef.current = { ...toolTimingsRef.current, ...content.tool_timings }
      }
    }

    if (obj.clear_history) {
      clearHistoryRef.current = true
      setQuestion(lastQuestionRef.current)
      setMapData(null)
      setStreamedOutput('')
      setLiveMessage('')
      setPatternSaved(false)
      setError(obj.error || 'Conversation history was corrupt and has been cleaned. Please retry.')
    }
  }

  async function savePattern() {
    try {
      setPatternSaving(true)
      const res = await fetch(`${API_URL}/pattern/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, session_id: sessionId, history }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to save pattern')
      setPatternSaved(true)
    } catch (e) { setError(String(e)) }
    finally { setPatternSaving(false) }
  }

  async function sendFeedback(feedback) {
    if (feedbackGiven !== null) return
    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, session_id: sessionId, feedback, history }),
      })
      if (!res.ok) throw new Error('Failed to record feedback')
      setFeedbackGiven(feedback)
    } catch (e) { setError(String(e)) }
  }

  // Append a completed turn to the persistent transcript. The transcript is
  // append-only and independent of the server's history array, so the full
  // conversation stays scrollable even though history is trimmed for RAM.
  function commitTurn(userText) {
    let toolCalls = []
    let parsedAssistant = null
    const hist = finalHistoryRef.current
    if (Array.isArray(hist)) {
      const parsed = parseHistoryToTurns(hist)
      if (parsed.length > 0) {
        const last = parsed[parsed.length - 1]
        toolCalls = (last.toolCalls || []).map(tc => {
          const ms = toolTimingsRef.current[tc.toolUseId]
          return typeof ms === 'number' ? { ...tc, timeMs: ms } : tc
        })
        parsedAssistant = last.assistantText
      }
    }
    // Prefer the authoritative final answer captured from content.text; fall back
    // to whatever the parsed history yielded.
    const finalText = finalAnswerTextRef.current && finalAnswerTextRef.current.trim()
      ? stripJsonDataBlock(finalAnswerTextRef.current.trim())
      : parsedAssistant
    setTranscript(prev => [...prev, {
      id: prev.length,
      userText,
      assistantText: finalText || null,
      toolCalls,
      reasoningSteps: [...frozenReasoningRef.current],
      mapData: currentMapDataRef.current,
    }])
  }

  async function submitQuestion(overrideInput, historyOverride, sessionOverride) {
    const inputToSend = overrideInput !== undefined ? String(overrideInput) : question
    const historyToSend = historyOverride !== undefined ? historyOverride : history
    const sessionToSend = sessionOverride !== undefined ? sessionOverride : sessionId
    lastQuestionRef.current = inputToSend
    if (overrideInput === undefined) setQuestion('')
    setLoading(true)
    setError(null)
    setStreamedOutput('')
    setStatus(null)
    setLiveMessage('')
    setPatternSaved(false)
    setFeedbackGiven(null)
    setReasoningSteps([])
    frozenReasoningRef.current = []
    currentMapDataRef.current = null
    finalHistoryRef.current = null
    finalAnswerTextRef.current = ''
    toolTimingsRef.current = {}
    clearHistoryRef.current = false
    setMapData(null)
    if (!inputToSend.startsWith('Hi, my username is')) {
      setPendingQuestion(inputToSend)
    }

    try {
      const res = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: inputToSend, history: historyToSend, user_id: userId, username, session_id: sessionToSend, function_stage: functionStage }),
      })

      if (!res.ok && res.status !== 404) {
        let errMsg = `HTTP ${res.status}`
        try { const d = await res.json(); errMsg = d.error || JSON.stringify(d) } catch {}
        throw new Error(errMsg)
      }

      if (res.ok && res.body) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() || ''
          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed) continue
            const payload = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed
            consumeStreamPayload(payload)
          }
        }
        if (buf.trim()) {
          const payload = buf.trim().startsWith('data:') ? buf.trim().slice(5).trim() : buf.trim()
          consumeStreamPayload(payload)
        }
      } else {
        const res2 = await fetch(`${API_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: inputToSend, history: historyToSend, user_id: userId, username, session_id: sessionToSend, function_stage: functionStage }),
        })
        const data = await res2.json()
        if (!res2.ok) throw new Error(data.error || 'API error')
        setHistory(data.history || history)
        finalHistoryRef.current = data.history || history
        const c = data.content
        if (c && typeof c === 'object' && typeof c.text === 'string' && c.text.trim()) {
          finalAnswerTextRef.current = c.text
        }
        if (c && typeof c === 'object' && c.tool_timings && typeof c.tool_timings === 'object') {
          toolTimingsRef.current = { ...toolTimingsRef.current, ...c.tool_timings }
        }
      }

      // Persist this turn in the append-only transcript unless the backend told
      // us the history was corrupt (that path retries and will commit on success).
      if (!clearHistoryRef.current) {
        commitTurn(inputToSend)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
      setPendingQuestion(null)
    }
  }

  function clearHistory() {
    // Finalize the old session's checkpoint (fold in any un-checkpointed tail) before
    // abandoning it — it stays durable and recallable later by browser (user_id).
    const oldSessionId = sessionId
    const oldHistory = finalHistoryRef.current || history
    if (oldSessionId && oldHistory && oldHistory.length) {
      fetch(`${API_URL}/checkpoint/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, username, session_id: oldSessionId, history: oldHistory }),
      }).catch(() => {})
    }
    const newSessionId = crypto.randomUUID()
    localStorage.setItem('mcp_session_id', newSessionId)
    setError(null)
    setHistory(null)
    setTranscript([])
    setSessionId(newSessionId)
    setStatus(null)
    setLiveMessage('')
    setStreamedOutput('')
    setMapData(null)
    setPatternSaved(false)
    setFeedbackGiven(null)
    setReasoningSteps([])
    setPendingQuestion(null)
    frozenReasoningRef.current = []
    currentMapDataRef.current = null
    finalHistoryRef.current = null
    finalAnswerTextRef.current = ''
    clearHistoryRef.current = false
    submitQuestion(SESSION_GREETING(username), null, newSessionId)
  }

  async function toggleFunctionStage() {
    const next = functionStage === 'prod' ? 'dev' : 'prod'
    setStageSwitching(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/function_stage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: next }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error || 'Failed to switch function stage')
      const applied = data.function_stage || next
      setFunctionStage(applied)
      localStorage.setItem('mcp_function_stage', applied)
    } catch (e) { setError(String(e)) }
    finally { setStageSwitching(false) }
  }

  async function fetchMcpConfig(regenerate = false) {
    setMcpModalOpen(true)
    if (regenerate) {
      setRegeneratingToken(true)
    } else {
      setMcpLoading(true)
    }
    setMcpError(null)
    setCopiedConfig(false)
    setCopiedToken(false)
    try {
      const res = await fetch(`${API_URL}/mcp_config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, regenerate }),
      })
      const data = await res.json()
      if (!res.ok || data.status === 'error') {
        throw new Error(data.error || 'Failed to retrieve MCP configuration')
      }
      setMcpConfigData(data)
    } catch (e) {
      setMcpError(String(e.message || e))
    } finally {
      setMcpLoading(false)
      setRegeneratingToken(false)
    }
  }

  function copyToClipboard(text, type) {
    if (!text) return
    const onSuccess = () => {
      if (type === 'config') {
        setCopiedConfig(true)
        setTimeout(() => setCopiedConfig(false), 2000)
      } else if (type === 'token') {
        setCopiedToken(true)
        setTimeout(() => setCopiedToken(false), 2000)
      }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
        fallbackCopy(text, onSuccess)
      })
    } else {
      fallbackCopy(text, onSuccess)
    }
  }

  function fallbackCopy(text, onSuccess) {
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.left = '-999999px'
      ta.style.top = '-999999px'
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      onSuccess()
    } catch (err) {
      console.error('Failed to copy text', err)
    }
  }

  // Greeting turn: hide the user bubble but show the AI's welcome response.
  const isGreeting = (t) => t.userText.startsWith('Hi, my username is')
  const visibleTurns = transcript
  const lastTurn = visibleTurns[visibleTurns.length - 1]

  return (
    <div className="chat-root">
      {/* Header */}
      <header className="chat-header">
        <img src="/leaflogo.png" alt="Logo" style={{ height: 44, width: 'auto' }} />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#001E2B' }}>MongoDB Atlas MCP AI Demo</h1>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className="btn-secondary"
            onClick={() => fetchMcpConfig(false)}
            title="View your MCP JWT token and copy VS Code configuration"
            style={{ fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            🔑 Get MCP Config
          </button>
          <button
            className="btn-secondary"
            onClick={toggleFunctionStage}
            disabled={stageSwitching}
            title="Toggle which dynamic query functions the model can see. 'dev' surfaces in-development functions; 'prod' shows only published ones."
            style={functionStage === 'dev'
              ? { fontSize: 13, backgroundColor: '#FF5C35', borderColor: '#FF5C35', color: '#fff', display: 'inline-flex', alignItems: 'center', gap: 6, cursor: stageSwitching ? 'wait' : 'pointer', opacity: stageSwitching ? 0.7 : 1 }
              : { fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 6, cursor: stageSwitching ? 'wait' : 'pointer', opacity: stageSwitching ? 0.7 : 1 }}
          >
            {stageSwitching && <span className="mcb-spinner" aria-hidden="true" />}
            {stageSwitching ? 'Switching…' : `Functions: ${functionStage === 'dev' ? 'Dev' : 'Prod'}`}
          </button>
          <label htmlFor="username-input" style={{ fontSize: 13, color: '#555' }}>User:</label>
          <input
            id="username-input"
            type="text"
            value={usernameDraft}
            disabled={authLocked}
            onChange={e => setUsernameDraft(e.target.value)}
            onKeyDown={e => {
              if (authLocked) return
              if (e.key === 'Enter') { setUsername(usernameDraft); setCookie('mcp_username', usernameDraft) }
            }}
            style={{ fontSize: 13, padding: '4px 8px', border: '1px solid #ccc', borderRadius: 6, width: 130, backgroundColor: authLocked ? '#f0f0f0' : '#fff', cursor: authLocked ? 'not-allowed' : 'text' }}
          />
          {!authLocked && (
            <button className="btn-secondary" onClick={() => { setUsername(usernameDraft); setCookie('mcp_username', usernameDraft) }}>
              Save
            </button>
          )}
        </div>
      </header>

      {/* Chat body */}
      <main className="chat-body" ref={chatBodyRef}>
        {visibleTurns.filter(t => !isGreeting(t) || t.assistantText).length === 0 && !loading && (
          <div style={{ textAlign: 'center', color: '#aaa', marginTop: 60, fontSize: 14 }}>
            Ask anything — your conversation will appear here.
          </div>
        )}

        {/* Initial load / hidden-greeting spinner: shown while the first response
            is being generated (the greeting sets no pendingQuestion bubble). */}
        {loading && !pendingQuestion && visibleTurns.filter(t => !isGreeting(t) || t.assistantText).length === 0 && (
          <div className="mcb-loading">
            <div className="mcb-spinner-lg" aria-hidden="true" />
            <div className="mcb-loading-text">Warming up your assistant…</div>
          </div>
        )}

        {visibleTurns.map((turn, idx) => {
          const isLast = idx === visibleTurns.length - 1
          return (
            <ChatMessage
              key={turn.id}
              userText={isGreeting(turn) ? null : turn.userText}
              assistantText={turn.assistantText}
              toolCalls={turn.toolCalls}
              reasoningSteps={turn.reasoningSteps || []}
              mapData={turn.mapData || (isLast && !loading ? mapData : null)}
              isStreaming={false}
              modelId={modelId}
            />
          )
        })}

        {/* In-progress bubble */}
        {loading && pendingQuestion && (
          <>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
              <div style={{
                maxWidth: '75%', background: '#00ED64', color: '#001E2B',
                borderRadius: '18px 18px 4px 18px', padding: '10px 16px',
                fontSize: 14, fontWeight: 500,
              }}>
                {pendingQuestion}
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 16 }}>
              <div style={{ maxWidth: '85%' }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4, marginLeft: 2 }}>
                  <span style={{ fontWeight: 600, color: '#00684A' }}>● Atlas AI</span>
                </div>
                <div className="streaming-bubble">
                  {status && <div className="status-badge">{status}</div>}
                  {streamedOutput
                    ? <pre ref={liveLogRef} style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 12, maxHeight: 200, overflow: 'auto' }}>{streamedOutput}</pre>
                    : <span style={{ color: '#aaa', fontSize: 13 }}>⏳ Processing...</span>
                  }
                </div>
              </div>
            </div>
          </>
        )}

        {error && (
          <div style={{ color: '#c0392b', background: '#fff0f0', border: '1px solid #f0c0bb', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
            ❌ {error}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="chat-footer">
        {lastTurn?.assistantText && !loading && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
            <button className="btn-secondary" onClick={savePattern} disabled={patternSaved || patternSaving} style={{ fontSize: 13 }}>
              {patternSaved ? '✅ Saved' : patternSaving ? '⏳' : '👍 Save pattern'}
            </button>
            <button
              className="btn-secondary"
              onClick={() => sendFeedback('negative')}
              disabled={feedbackGiven !== null}
              style={{ fontSize: 13, color: feedbackGiven === 'negative' ? '#c0392b' : undefined }}
            >
              {feedbackGiven === 'negative' ? '🚩 Flagged' : '👎 Flag'}
            </button>
          </div>
        )}
        <div className="chat-input-row">
          <textarea
            className="chat-textarea"
            placeholder="Ask anything… (Enter to send, Shift+Enter for new line)"
            value={question}
            rows={2}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!loading) submitQuestion() }
            }}
          />
          <button className="btn-send" onClick={() => submitQuestion()} disabled={loading}>
            {loading ? '⏳' : 'Send'}
          </button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button className="btn-secondary" onClick={clearHistory} disabled={loading}>Clear chat</button>
        </div>
      </footer>

      {/* VS Code MCP Configuration Modal */}
      {mcpModalOpen && (
        <div className="mcp-modal-backdrop" onClick={() => setMcpModalOpen(false)}>
          <div className="mcp-modal-dialog" onClick={e => e.stopPropagation()}>
            <div className="mcp-modal-header">
              <h2>
                <span role="img" aria-label="key">🔑</span> VS Code MCP Configuration
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mcp-badge">User: {username}</span>
                <button
                  className="mcp-modal-close"
                  onClick={() => setMcpModalOpen(false)}
                  title="Close"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="mcp-modal-body">
              {mcpLoading ? (
                <div style={{ textAlign: 'center', padding: '40px 0', color: '#666' }}>
                  <span className="mcb-spinner" style={{ width: 24, height: 24, borderWidth: 3, borderColor: '#00ED64', borderRightColor: 'transparent' }} />
                  <div style={{ marginTop: 12 }}>Loading MCP credentials for {username}…</div>
                </div>
              ) : mcpError ? (
                <div style={{ color: '#c0392b', background: '#fff0f0', border: '1px solid #f0c0bb', borderRadius: 8, padding: '12px 16px', marginBottom: 12 }}>
                  ❌ {mcpError}
                </div>
              ) : mcpConfigData ? (
                <>
                  <p style={{ margin: '0 0 14px', color: '#555', lineHeight: 1.5 }}>
                    Copy this configuration into your workspace <code>.vscode/mcp.json</code> or VS Code settings to connect GitHub Copilot / VS Code MCP clients directly to this MongoDB server:
                  </p>

                  {/* VS Code Config JSON */}
                  <div className="mcp-section">
                    <div className="mcp-section-header">
                      <span className="mcp-section-title">VS Code Configuration (<code>.vscode/mcp.json</code>)</span>
                      <button
                        className={`mcp-copy-button ${copiedConfig ? 'copied' : ''}`}
                        onClick={() => copyToClipboard(mcpConfigData.vscode_config_json, 'config')}
                        title="Copy configuration JSON"
                      >
                        {copiedConfig ? '✅ Copied!' : '📋 Copy Config'}
                      </button>
                    </div>
                    <div className="mcp-code-container">
                      <pre className="mcp-code-block">{mcpConfigData.vscode_config_json}</pre>
                    </div>
                  </div>

                  {/* Bearer Token */}
                  <div className="mcp-section">
                    <div className="mcp-section-header">
                      <span className="mcp-section-title">Bearer JWT Token</span>
                      <button
                        className={`mcp-copy-button ${copiedToken ? 'copied' : ''}`}
                        onClick={() => copyToClipboard(mcpConfigData.token, 'token')}
                        title="Copy raw JWT token"
                      >
                        {copiedToken ? '✅ Copied!' : '📋 Copy Token'}
                      </button>
                    </div>
                    <pre className="mcp-token-block">{mcpConfigData.token}</pre>
                  </div>

                  {/* Metadata info row */}
                  <div className="mcp-info-row">
                    <div className="mcp-info-item">
                      <span className="mcp-info-label">Server Root</span>
                      <span className="mcp-info-value">{mcpConfigData.mcp_root}</span>
                    </div>
                    <div className="mcp-info-item">
                      <span className="mcp-info-label">Endpoints ({mcpConfigData.endpoints?.length || 0})</span>
                      <span className="mcp-info-value">{mcpConfigData.endpoints?.join(', ')}</span>
                    </div>
                    <div className="mcp-info-item">
                      <span className="mcp-info-label">Scopes</span>
                      <span className="mcp-info-value">{Array.isArray(mcpConfigData.scope) ? mcpConfigData.scope.join(', ') : String(mcpConfigData.scope)}</span>
                    </div>
                  </div>
                </>
              ) : null}
            </div>

            <div className="mcp-modal-footer">
              <button
                className="btn-secondary"
                onClick={() => fetchMcpConfig(true)}
                disabled={mcpLoading || regeneratingToken}
                title="Generate a new token for this user (replaces existing key in MongoDB)"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13, color: '#c0392b', borderColor: '#f0c0bb' }}
              >
                {regeneratingToken && <span className="mcb-spinner" aria-hidden="true" />}
                {regeneratingToken ? 'Generating…' : '🔄 Generate New Token'}
              </button>
              <button
                className="btn-secondary"
                onClick={() => setMcpModalOpen(false)}
                style={{ fontSize: 13, padding: '6px 16px' }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
