import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ExpandableTile from './ExpandableTile'
import JsonDataRenderer from './JsonDataRenderer'

/**
 * Renders one conversation turn: user bubble + AI bubble + detail tiles.
 */

/** Deeply parse a value that may be a JSON string (possibly double-encoded). */
function deepParse(val) {
  if (typeof val !== 'string') return val
  let v = val
  // Up to 3 levels of JSON string unwrapping
  for (let i = 0; i < 3; i++) {
    try {
      const parsed = JSON.parse(v)
      if (typeof parsed === 'string') { v = parsed; continue }
      return parsed
    } catch { break }
  }
  return v
}

/** Syntax-highlight a JSON string token by token. */
function highlight(str) {
  // Replace JSON tokens with colored spans
  return str.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      if (/^"/.test(match)) {
        if (/:$/.test(match)) return `<span style="color:#7ec8e3">${match}</span>` // key
        return `<span style="color:#a8d8a8">${match}</span>` // string value
      }
      if (/true|false/.test(match)) return `<span style="color:#f0a500">${match}</span>`
      if (/null/.test(match)) return `<span style="color:#e07070">${match}</span>`
      return `<span style="color:#c9c">${match}</span>` // number
    }
  )
}

function PrettyJson({ data }) {
  const parsed = deepParse(data)
  const text = typeof parsed === 'string'
    ? parsed
    : JSON.stringify(parsed, null, 2)
  const html = highlight(text)
  return (
    <pre
      style={{
        margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        fontSize: 12, maxHeight: 420, overflowY: 'auto',
        background: '#001E2B', color: '#cdd', padding: 12, borderRadius: 6,
        lineHeight: 1.5,
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

/** Collapsible JSON node tree. */
function JsonNode({ data, depth = 0 }) {
  const [open, setOpen] = useState(depth < 2)

  if (data === null) return <span style={{ color: '#e07070' }}>null</span>
  if (typeof data === 'boolean') return <span style={{ color: '#f0a500' }}>{String(data)}</span>
  if (typeof data === 'number') return <span style={{ color: '#c9c' }}>{data}</span>
  if (typeof data === 'string') return <span style={{ color: '#a8d8a8' }}>"{data}"</span>

  const isArray = Array.isArray(data)
  const entries = isArray ? data.map((v, i) => [i, v]) : Object.entries(data)
  const isEmpty = entries.length === 0

  const bracket = isArray ? ['[', ']'] : ['{', '}']
  const indent = { marginLeft: 16 }

  if (isEmpty) return <span style={{ color: '#aaa' }}>{bracket[0]}{bracket[1]}</span>

  return (
    <span>
      <span
        style={{ cursor: 'pointer', userSelect: 'none', color: '#7ec8e3', fontWeight: 600 }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? '▾' : '▸'} {bracket[0]}
      </span>
      {!open && (
        <span style={{ color: '#888', cursor: 'pointer' }} onClick={() => setOpen(true)}>
          {isArray ? ` ${entries.length} items` : ` ${entries.length} keys`} {bracket[1]}
        </span>
      )}
      {open && (
        <div style={indent}>
          {entries.map(([k, v], i) => (
            <div key={k} style={{ marginBottom: 1 }}>
              {!isArray && <span style={{ color: '#7ec8e3' }}>"{k}"</span>}
              {!isArray && <span style={{ color: '#aaa' }}>: </span>}
              <JsonNode data={v} depth={depth + 1} />
              {i < entries.length - 1 && <span style={{ color: '#555' }}>,</span>}
            </div>
          ))}
        </div>
      )}
      {open && <span style={{ color: '#7ec8e3' }}>{bracket[1]}</span>}
    </span>
  )
}

function JsonViewer({ data }) {
  const [mode, setMode] = useState('tree')
  const parsed = deepParse(data)
  const isObj = parsed !== null && typeof parsed === 'object'

  return (
    <div>
      {isObj && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
          {['tree', 'raw'].map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                fontSize: 11, padding: '2px 8px', borderRadius: 10, cursor: 'pointer',
                border: '1px solid #00ED64',
                background: mode === m ? '#00ED64' : 'transparent',
                color: mode === m ? '#001E2B' : '#00ED64',
                fontWeight: 600,
              }}
            >
              {m}
            </button>
          ))}
        </div>
      )}
      {mode === 'tree' && isObj ? (
        <div style={{
          background: '#001E2B', color: '#cdd', padding: 12, borderRadius: 6,
          fontSize: 12, lineHeight: 1.6, maxHeight: 420, overflowY: 'auto',
          fontFamily: "'SFMono-Regular', Consolas, monospace",
        }}>
          <JsonNode data={parsed} depth={0} />
        </div>
      ) : (
        <PrettyJson data={data} />
      )}
    </div>
  )
}

/** Extract a friendly display name from a Bedrock model ID like "global.anthropic.claude-sonnet-4-6" */
function friendlyModelName(modelId) {
  if (!modelId) return 'Atlas AI'
  // Strip cross-region prefix (e.g. "global.", "us.")
  const base = modelId.replace(/^[a-z-]+\./, '').replace(/^[a-z-]+\./, '')
  // Extract just the model slug after provider prefix
  const parts = base.split('.')
  const slug = parts[parts.length - 1] || parts[0]
  // Capitalise and clean up hyphens
  return slug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export default function ChatMessage({
  userText,
  assistantText,
  toolCalls = [],
  reasoningSteps = [],
  mapData = null,
  isStreaming = false,
  modelId = '',
}) {
  const hasTools = toolCalls.length > 0
  const hasReasoning = reasoningSteps.length > 0
  const hasMap = !!mapData

  return (
    <div style={{ marginBottom: 24 }}>
      {/* User bubble — omitted for greeting turn */}
      {userText && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
          <div style={{
            maxWidth: '75%',
            background: '#00ED64',
            color: '#001E2B',
            borderRadius: '18px 18px 4px 18px',
            padding: '10px 16px',
            fontSize: 14,
            fontWeight: 500,
            lineHeight: 1.5,
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {userText}
          </div>
        </div>
      )}

      {/* AI bubble */}
      {(assistantText || isStreaming) && (
        <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
          <div style={{ maxWidth: '85%' }}>
            {/* Avatar + name */}
            <div style={{ fontSize: 11, color: '#888', marginBottom: 4, marginLeft: 2 }}>
              <span style={{ fontWeight: 600, color: '#00684A' }}>● {friendlyModelName(modelId)}</span>
            </div>

            <div style={{
              background: '#fff',
              border: '1px solid #e8f5ee',
              borderRadius: '4px 18px 18px 18px',
              padding: '12px 16px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
            }}>
              {assistantText ? (
                <div className="markdown-content" style={{ fontSize: 14, lineHeight: 1.6 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{assistantText}</ReactMarkdown>
                </div>
              ) : (
                <div style={{ color: '#aaa', fontSize: 13 }}>⏳ Thinking...</div>
              )}
            </div>

            {/* Map data */}
            {hasMap && (
              <div style={{ marginTop: 8 }}>
                <JsonDataRenderer jsonData={mapData} />
              </div>
            )}

            {/* Tile chips row */}
            {!isStreaming && (hasTools || hasReasoning || hasMap) && (
              <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>

                {hasTools && (
                  <ExpandableTile icon="🔧" label="Tools called" count={toolCalls.length}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {toolCalls.map((tc, i) => (
                        <div key={i}>
                          <div style={{ fontWeight: 600, color: '#001E2B', marginBottom: 4 }}>
                            {i + 1}. <code style={{ background: '#e8f5ee', padding: '1px 6px', borderRadius: 4 }}>{tc.name}</code>
                          </div>
                          <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>Input:</div>
                          <JsonViewer data={tc.input} />
                        </div>
                      ))}
                    </div>
                  </ExpandableTile>
                )}

                {hasTools && (
                  <ExpandableTile icon="📦" label="Tool responses" count={toolCalls.filter(tc => tc.result !== undefined).length}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {toolCalls.map((tc, i) => (
                        <div key={i}>
                          <div style={{ fontWeight: 600, color: '#001E2B', marginBottom: 4 }}>
                            <code style={{ background: '#e8f5ee', padding: '1px 6px', borderRadius: 4 }}>{tc.name}</code>
                          </div>
                          {tc.result !== undefined ? (
                            <JsonViewer data={tc.result} />
                          ) : (
                            <span style={{ color: '#aaa', fontSize: 12 }}>No response captured</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </ExpandableTile>
                )}

                {hasReasoning && (
                  <ExpandableTile icon="🧠" label="Processing log" count={reasoningSteps.length}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {reasoningSteps.map((step, i) => {
                        const rawMsg = typeof step === 'string' ? step : step.message
                        const sts = typeof step === 'string' ? '' : step.status
                        const msg = sts === 'LLM Reasoning' && rawMsg && rawMsg.length > 100
                          ? rawMsg.slice(0, 100) + '…'
                          : rawMsg
                        return (
                          <div key={i} style={{
                            fontSize: 12,
                            padding: '3px 0',
                            borderBottom: i < reasoningSteps.length - 1 ? '1px solid #eee' : 'none',
                            display: 'flex',
                            gap: 8,
                            alignItems: 'flex-start',
                          }}>
                            {sts && (
                              <span style={{
                                flexShrink: 0,
                                fontSize: 10,
                                fontWeight: 700,
                                color: '#fff',
                                background: sts === 'LLM Reasoning' ? '#00684A' : sts.includes('Error') ? '#c0392b' : '#334',
                                borderRadius: 8,
                                padding: '1px 6px',
                                marginTop: 1,
                              }}>
                                {sts}
                              </span>
                            )}
                            <span style={{ color: '#333', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{msg}</span>
                          </div>
                        )
                      })}
                    </div>
                  </ExpandableTile>
                )}

              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
