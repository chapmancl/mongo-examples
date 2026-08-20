import MapResultsDisplay, { canDisplayAsMap } from './MapResultsDisplay.jsx';
import MemoryGraphDisplay from './MemoryGraphDisplay.jsx';

/**
 * Dispatches a jsondata object to the correct React component
 * based on the jsonDataType field. Add new cases here as new data types are supported.
 */
export default function JsonDataRenderer({ jsonData }) {
  if (!jsonData) return null;

  switch (jsonData.jsonDataType) {
    case 'geospatial_scatter':
      return <MapResultsDisplay mapData={jsonData} />;
    case 'memory_graph':
      return <MemoryGraphDisplay graphData={jsonData} />;
    case 'multi':
      // Multiple bypassed tool results in one turn — render each independently
      // (map/graph/table) via the same dispatch. Kept separate, not merged.
      return (
        <>
          {(jsonData.items || []).map((it, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              {it.tool && (
                <div style={{ fontSize: 12, color: '#00684A', fontWeight: 600, marginBottom: 4 }}>
                  {it.domain ? `${it.domain}: ` : ''}{it.tool}
                </div>
              )}
              <JsonDataRenderer jsonData={it.data} />
            </div>
          ))}
        </>
      );
    default: {
      // Try to auto-detect a mappable format before falling back to raw JSON.
      if (canDisplayAsMap(jsonData)) {
        return <MapResultsDisplay mapData={jsonData} />;
      }

      return (
        <div style={{ padding: 12, background: '#f9f9f9', border: '1px solid #ddd', borderRadius: 4 }}>
          {jsonData.jsonDataType && (
            <div style={{ marginBottom: 8 }}>
              <strong>Unsupported data type:</strong> {jsonData.jsonDataType}
            </div>
          )}
          <pre
            style={{
              margin: 0,
              padding: 12,
              background: '#111827',
              color: '#e5e7eb',
              borderRadius: 6,
              overflowX: 'auto',
              fontSize: 12,
              lineHeight: 1.4,
            }}
          >
            <code>{JSON.stringify(jsonData, null, 2)}</code>
          </pre>
        </div>
      );
    }
  }
}
