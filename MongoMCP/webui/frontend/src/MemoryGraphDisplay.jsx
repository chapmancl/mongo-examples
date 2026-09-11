import React, { useRef, useCallback, useState, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

// Hop-depth colour palette (index = hop depth)
const HOP_COLORS = ['#3b82f6', '#f97316', '#22c55e', '#a78bfa', '#ec4899']
const HOP_LABELS = ['Seed (hop 0)', 'Hop 1', 'Hop 2', 'Hop 3', 'Hop 4']

// Edge type colours
const EDGE_COLOR = { direct: '#94a3b8', entity: '#16a34a', tag: '#f59e0b' }

function hopColor(hopDepth) {
  return HOP_COLORS[hopDepth] ?? '#6b7280'
}

function nodeRadius(importance) {
  const imp = typeof importance === 'number' ? Math.max(0, Math.min(1, importance)) : 0.5
  return imp * 10 + 5 // 5–15 px
}

/**
 * Renders a jsonDataType="memory_graph" payload as a force-directed graph.
 *
 * Expected graphData shape:
 * {
 *   jsonDataType: "memory_graph",
 *   nodes: [ { id, label, hop_depth, memory_type, importance, score, tags, hydrated, content_preview } ],
 *   edges: [ { source, target, relation, hop } ],
 *   query, paths_used, count, graph_neighbor_count, fallback_used, depth, scope
 * }
 */
export default function MemoryGraphDisplay({ graphData }) {
  const fgRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 700, height: 480 })
  const [hoveredNode, setHoveredNode] = useState(null)

  const btnStyle = {
    padding: '3px 10px',
    fontSize: 13,
    fontWeight: 600,
    borderRadius: 4,
    border: '1px solid #d1d5db',
    background: '#fff',
    color: '#374151',
    cursor: 'pointer',
    lineHeight: 1.5,
  }

  const {
    nodes = [],
    edges = [],
    query,
    paths_used,
    count,
    graph_neighbor_count,
    fallback_used,
    depth,
    scope,
  } = graphData || {}

  // Responsive width — debounced, ignores sub-5px jitter to prevent
  // ResizeObserver → dimension change → node reposition → fg2dData reinit loop.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let timer = null
    const ro = new ResizeObserver(entries => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        for (const entry of entries) {
          const w = Math.floor(entry.contentRect.width)
          if (w > 0) setDimensions(d => {
            if (Math.abs(d.width - Math.max(300, w)) < 5) return d
            return { ...d, width: Math.max(300, w) }
          })
        }
      }, 200)
    })
    ro.observe(el)
    return () => { ro.disconnect(); clearTimeout(timer) }
  }, [])

  // Static concentric ring layout: hop_depth 0 = centre, each hop = an outer ring.
  // Setting fx/fy fixes node positions so D3 never simulates them — zero CPU, zero freeze.
  const graphNodes = React.useMemo(() => {
    const cx = dimensions.width / 2
    const cy = dimensions.height / 2
    // RING_RADII[0]=0 means a single hop-0 node lands at center.
    // Multiple nodes at the same hop use the larger fallback below.
    const RING_RADII = [0, 130, 230, 320, 400]
    const HOP_MULTI_RADII = [90, 130, 230, 320, 400] // used when >1 node shares a hop

    // Group by hop depth
    const byHop = {}
    nodes.forEach(n => {
      const hop = n.hop_depth ?? 0
      if (!byHop[hop]) byHop[hop] = []
      byHop[hop].push(n)
    })

    return nodes.map(n => {
      const hop = n.hop_depth ?? 0
      const group = byHop[hop]
      const idx = group.indexOf(n)
      // Use a small ring when multiple nodes share a hop so they don't stack.
      const r = group.length === 1
        ? (RING_RADII[hop] ?? hop * 100)
        : (HOP_MULTI_RADII[hop] ?? Math.max(90, hop * 100))
      const angle = group.length === 1
        ? -Math.PI / 2  // single node at top
        : (idx / group.length) * 2 * Math.PI - Math.PI / 2
      return {
        ...n,
        fx: r === 0 ? cx : cx + r * Math.cos(angle),
        fy: r === 0 ? cy : cy + r * Math.sin(angle),
      }
    })
  }, [nodes, dimensions.width, dimensions.height])

  // Direct explicit edges (related_docs links from backend)
  const directLinks = React.useMemo(
    () => edges.map((e, i) => ({ ...e, _type: 'direct', _id: `${e.source}-${e.target}-${i}` })),
    [edges]
  )

  // All edges: direct (related_docs) > entity (shared entities) > tag (shared tags)
  // Priority per pair: only one edge shown, highest-priority type wins.
  const allGraphLinks = React.useMemo(() => {
    // Build a set of node IDs for quick lookup
    const nodeIds = new Set(graphNodes.map(n => String(n.id)))
    // Track best link per ordered pair (sorted IDs as key)
    const bestByPair = new Map()
    const PRIORITY = { direct: 0, entity: 1, tag: 2 }

    const consider = (link) => {
      const src = String(link.source)
      const tgt = String(link.target)
      if (!nodeIds.has(src) || !nodeIds.has(tgt)) return // skip dangling edges
      const key = [src, tgt].sort().join('||')
      const existing = bestByPair.get(key)
      if (!existing || PRIORITY[link._type] < PRIORITY[existing._type]) {
        bestByPair.set(key, link)
      }
    }

    // Direct links (related_docs from backend) — highest priority
    directLinks.forEach(l => consider(l))

    // Infer entity/tag edges between all node pairs
    for (let i = 0; i < graphNodes.length; i++) {
      for (let j = i + 1; j < graphNodes.length; j++) {
        const a = graphNodes[i]
        const b = graphNodes[j]
        const sharedEntities = (a.entities || []).filter(e => (b.entities || []).includes(e))
        const sharedTags = (a.tags || []).filter(t => (b.tags || []).includes(t))
        if (sharedEntities.length > 0) {
          consider({
            source: a.id, target: b.id,
            relation: `entity: ${sharedEntities.slice(0, 3).join(', ')}`,
            _type: 'entity',
            _id: `${a.id}||${b.id}||entity`,
          })
        }
        if (sharedTags.length > 0) {
          consider({
            source: a.id, target: b.id,
            relation: `tag: ${sharedTags.slice(0, 3).join(', ')}`,
            _type: 'tag',
            _id: `${a.id}||${b.id}||tag`,
          })
        }
      }
    }
    return [...bestByPair.values()]
  }, [graphNodes, directLinks])

  const fg2dData = React.useMemo(
    () => ({ nodes: graphNodes, links: allGraphLinks }),
    [graphNodes, allGraphLinks]
  )

  // Fit graph after data changes — placed here so fg2dData is already declared (avoids TDZ).
  useEffect(() => {
    const timer = setTimeout(() => fgRef.current?.zoomToFit(400, 20), 120)
    return () => clearTimeout(timer)
  }, [fg2dData])

  // Custom canvas render for tag/entity edges (dashed lines)
  const linkCanvasObject = useCallback((link, ctx) => {
    const start = link.source
    const end = link.target
    if (typeof start !== 'object' || typeof end !== 'object') return
    const color = EDGE_COLOR[link._type] ?? EDGE_COLOR.direct
    ctx.save()
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.setLineDash(link._type === 'entity' ? [6, 3] : [3, 3])
    ctx.stroke()
    ctx.restore()
  }, [])

  // Zoom controls
  const handleZoomIn  = useCallback(() => {
    if (!fgRef.current) return
    fgRef.current.zoom(fgRef.current.zoom() * 1.5, 300)
  }, [])
  const handleZoomOut = useCallback(() => {
    if (!fgRef.current) return
    fgRef.current.zoom(fgRef.current.zoom() / 1.5, 300)
  }, [])
  const handleFit = useCallback(() => {
    fgRef.current?.zoomToFit(400, 20)
  }, [])

  // Custom canvas render for each node
  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const r = nodeRadius(node.importance)
    const color = hopColor(node.hop_depth ?? 0)
    const isHydrated = node.hydrated !== false
    const isHovered = hoveredNode && hoveredNode.id === node.id

    ctx.beginPath()
    ctx.arc(node.x, node.y, r + (isHovered ? 3 : 0), 0, 2 * Math.PI, false)

    if (isHydrated) {
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = isHovered ? '#fff' : 'rgba(0,0,0,0.25)'
      ctx.lineWidth = (isHovered ? 2.5 : 1.5) / globalScale
      ctx.setLineDash([])
      ctx.stroke()
    } else {
      // Stub: transparent fill + dashed border
      ctx.fillStyle = color + '33' // ~20% opacity
      ctx.fill()
      ctx.strokeStyle = color
      ctx.lineWidth = (isHovered ? 2.5 : 2) / globalScale
      ctx.setLineDash([5 / globalScale, 3 / globalScale])
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Node label below circle
    const label = node.label ?? String(node.id ?? '')
    const maxLen = 26
    const truncated = label.length > maxLen ? label.slice(0, maxLen - 1) + '…' : label
    const fontSize = Math.max(8, 11 / globalScale)
    ctx.font = `${fontSize}px sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillStyle = '#1f2937'
    ctx.fillText(truncated, node.x, node.y + r + 3 / globalScale)
  }, [hoveredNode])

  // Larger hit area so small nodes are still clickable
  const nodePointerAreaPaint = useCallback((node, color, ctx) => {
    const r = nodeRadius(node.importance)
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI, false)
    ctx.fill()
  }, [])

  const handleNodeHover = useCallback((node) => {
    setHoveredNode(node || null)
    if (containerRef.current) {
      containerRef.current.style.cursor = node ? 'pointer' : 'default'
    }
  }, [])

  // On drag end, fix the node at its new position so it stays put
  const handleNodeDragEnd = useCallback((node) => {
    node.fx = node.x
    node.fy = node.y
  }, [])

  if (!nodes.length) {
    return (
      <div style={{ padding: 16, color: '#6b7280', fontStyle: 'italic', fontSize: 14 }}>
        No memory graph nodes to display.
      </div>
    )
  }

  const maxHopDepth = Math.min(depth ?? 0, HOP_LABELS.length - 1)

  return (
    <div style={{ width: '100%', fontFamily: 'sans-serif' }}>

      {/* Header */}
      <div style={{ marginBottom: 8, display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>Memory Graph</span>
        {query && (
          <span style={{ fontSize: 13, color: '#6b7280' }}>"{query}"</span>
        )}
      </div>

      {/* Node legend */}
      <div style={{ display: 'flex', gap: '8px 20px', flexWrap: 'wrap', marginBottom: 4, fontSize: 12, alignItems: 'center' }}>
        {HOP_COLORS.slice(0, maxHopDepth + 1).map((c, i) => (
          <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              display: 'inline-block', width: 12, height: 12,
              borderRadius: '50%', background: c, flexShrink: 0
            }} />
            {HOP_LABELS[i]}
          </span>
        ))}
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            display: 'inline-block', width: 12, height: 12, borderRadius: '50%',
            border: '2px dashed #9ca3af', background: 'transparent', flexShrink: 0
          }} />
          Stub (not hydrated)
        </span>
      </div>

      {/* Edge legend */}
      <div style={{ display: 'flex', gap: '8px 20px', flexWrap: 'wrap', marginBottom: 8, fontSize: 12, alignItems: 'center' }}>
        <span style={{ color: '#6b7280', fontWeight: 600 }}>Edges:</span>
        {[['direct', '─── direct link'], ['entity', '╌╌╌ entity'], ['tag', '┄┄┄ tag']].map(([type, label]) => (
          <span key={type} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 20, height: 2, background: EDGE_COLOR[type], flexShrink: 0 }} />
            <span style={{ color: EDGE_COLOR[type] }}>{label}</span>
          </span>
        ))}
      </div>

      {/* Zoom controls */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 6, alignItems: 'center' }}>
        <button onClick={handleZoomIn}  style={btnStyle} title="Zoom in">＋</button>
        <button onClick={handleZoomOut} style={btnStyle} title="Zoom out">－</button>
        <button onClick={handleFit}     style={btnStyle} title="Fit graph">Fit</button>
        <span style={{ fontSize: 11, color: '#9ca3af', marginLeft: 4 }}>
          Scroll to zoom · drag to pan · drag nodes to reposition
        </span>
      </div>

      {/* Graph canvas */}
      <div
        ref={containerRef}
        style={{
          border: '1px solid #e5e7eb',
          borderRadius: 8,
          overflow: 'hidden',
          background: '#f8fafc',
          width: '100%',
        }}
      >
        <ForceGraph2D
          ref={fgRef}
          graphData={fg2dData}
          width={dimensions.width}
          height={dimensions.height}
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaPaint={nodePointerAreaPaint}
          onNodeHover={handleNodeHover}
          onNodeDragEnd={handleNodeDragEnd}
          linkLabel={link => link.relation || ''}
          linkDirectionalArrowLength={link => link._type === 'direct' ? 7 : 0}
          linkDirectionalArrowRelPos={0.85}
          linkColor={link => EDGE_COLOR[link._type] ?? EDGE_COLOR.direct}
          linkWidth={link => link._type === 'direct' ? 1.5 : 1}
          linkCanvasObjectMode={link => link._type !== 'direct' ? 'replace' : undefined}
          linkCanvasObject={linkCanvasObject}
          cooldownTicks={0}
          d3AlphaDecay={1}
          nodeRelSize={1}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
      </div>

      {/* Hovered node detail panel */}
      <div style={{
        marginTop: 8,
        minHeight: 56,
        padding: '8px 14px',
        background: hoveredNode ? '#1e293b' : '#f1f5f9',
        color: hoveredNode ? '#f1f5f9' : '#94a3b8',
        borderRadius: 6,
        fontSize: 12,
        transition: 'background 0.15s ease, color 0.15s ease',
        lineHeight: 1.6,
      }}>
        {hoveredNode ? (
          <>
            <div style={{ fontWeight: 700, marginBottom: 2, fontSize: 13 }}>
              {hoveredNode.label ?? hoveredNode.id}
            </div>
            <div style={{ display: 'flex', gap: '4px 20px', flexWrap: 'wrap', opacity: 0.85 }}>
              {hoveredNode.memory_type && <span>Type: <strong>{hoveredNode.memory_type}</strong></span>}
              {typeof hoveredNode.importance === 'number' && (
                <span>Importance: <strong>{hoveredNode.importance.toFixed(2)}</strong></span>
              )}
              {typeof hoveredNode.score === 'number' && (
                <span>Score: <strong>{hoveredNode.score.toFixed(3)}</strong></span>
              )}
              {hoveredNode.hop_depth != null && (
                <span>Hop: <strong>{hoveredNode.hop_depth}</strong></span>
              )}
              {hoveredNode.hydrated === false && (
                <span style={{ color: '#fbbf24' }}>stub</span>
              )}
            </div>
            {hoveredNode.entities?.length > 0 && (
              <div style={{ marginTop: 2, opacity: 0.75 }}>
                Entities: {hoveredNode.entities.slice(0, 6).join(', ')}
              </div>
            )}
            {hoveredNode.tags?.length > 0 && (
              <div style={{ marginTop: 2, opacity: 0.75 }}>
                Tags: {hoveredNode.tags.slice(0, 8).join(', ')}
              </div>
            )}
            {hoveredNode.content_preview && (
              <div style={{ marginTop: 4, fontStyle: 'italic', opacity: 0.8 }}>
                "{hoveredNode.content_preview}"
              </div>
            )}
          </>
        ) : (
          <span>Hover a node for details</span>
        )}
      </div>

      {/* Metadata strip */}
      <div style={{
        marginTop: 6,
        padding: '8px 14px',
        background: '#f0f4f8',
        borderRadius: 6,
        fontSize: 12,
        display: 'flex',
        flexWrap: 'wrap',
        gap: '4px 20px',
        color: '#4b5563',
      }}>
        {count != null && <span><strong>Hydrated:</strong> {count}</span>}
        {graph_neighbor_count != null && <span><strong>Graph neighbors:</strong> {graph_neighbor_count}</span>}
        {paths_used != null && <span><strong>Paths:</strong> {paths_used}</span>}
        {depth != null && <span><strong>Depth:</strong> {depth}</span>}
        {scope != null && <span><strong>Scope:</strong> {scope}</span>}
        {fallback_used != null && (
          <span><strong>Fallback:</strong> {fallback_used ? 'yes' : 'no'}</span>
        )}
        <span><strong>Edges:</strong> {edges.length}</span>
      </div>
    </div>
  )
}
