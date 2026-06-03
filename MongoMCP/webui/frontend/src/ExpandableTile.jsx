import React, { useState } from 'react'

/**
 * Clickable chip that toggles an inline expanded panel.
 * Props: label, icon, count, children, defaultOpen
 */
export default function ExpandableTile({ label, icon, count, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div style={{ marginTop: 4 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 5,
          padding: '3px 10px',
          borderRadius: 20,
          border: '1px solid #00ED64',
          background: open ? '#00ED64' : 'transparent',
          color: open ? '#001E2B' : '#00684A',
          fontSize: 12,
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.15s ease',
          userSelect: 'none',
        }}
      >
        {icon && <span>{icon}</span>}
        {label}
        {count !== undefined && count > 0 && (
          <span style={{
            background: open ? '#001E2B' : '#00ED64',
            color: open ? '#00ED64' : '#001E2B',
            borderRadius: 10,
            padding: '0 6px',
            fontSize: 11,
            fontWeight: 700,
            marginLeft: 2,
          }}>
            {count}
          </span>
        )}
        <span style={{ fontSize: 10, marginLeft: 2 }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{
          marginTop: 6,
          padding: '10px 14px',
          background: '#f8fffe',
          border: '1px solid #d1f5e8',
          borderRadius: 8,
          fontSize: 13,
        }}>
          {children}
        </div>
      )}
    </div>
  )
}
