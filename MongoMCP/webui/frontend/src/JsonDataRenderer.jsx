import MapResultsDisplay from './MapResultsDisplay.jsx';

/**
 * Dispatches a jsondata object to the correct React component
 * based on the jsonDataType field. Add new cases here as new data types are supported.
 */
export default function JsonDataRenderer({ jsonData }) {
  if (!jsonData || !jsonData.jsonDataType) return null;

  switch (jsonData.jsonDataType) {
    case 'geospatial_scatter':
      return <MapResultsDisplay mapData={jsonData} />;
    default:
      return (
        <div style={{ padding: 12, background: '#f9f9f9', border: '1px solid #ddd', borderRadius: 4 }}>
          <div style={{ marginBottom: 8 }}>
            <strong>Unsupported data type:</strong> {jsonData.jsonDataType}
          </div>
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
