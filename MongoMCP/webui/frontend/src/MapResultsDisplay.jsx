import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

// Convert a GeoJSON FeatureCollection (Point features) into the internal mapData format.
function normalizeGeoJSON(geojson) {
  const markers = geojson.features
    .filter(f => f.geometry?.type === 'Point')
    .map((f, i) => {
      const [lng, lat] = f.geometry.coordinates;
      const props = f.properties ?? {};
      return {
        id: props.id ?? i,
        lat,
        lng,
        label: props.name ?? props.title ?? props.id ?? `Feature ${i}`,
        properties: props,   // kept for popup rendering
      };
    });
  return { markers };
}

function isGeoJSON(data) {
  return data?.type === 'FeatureCollection' && Array.isArray(data.features);
}

// Detect objects like { apartments: [{coordinates: [lng, lat], ...}] }
// Returns the first array of items that have a coordinates field, or null.
function findCoordinatesArray(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null;
  for (const key of Object.keys(data)) {
    const val = data[key];
    if (
      Array.isArray(val) &&
      val.length > 0 &&
      Array.isArray(val[0].coordinates) &&
      val[0].coordinates.length === 2
    ) {
      return val;
    }
  }
  return null;
}

function normalizeCoordinatesArray(items) {
  const markers = items.map((item, i) => {
    const [lng, lat] = item.coordinates;
    const { coordinates, ...rest } = item;  // exclude coordinates from popup props
    return {
      id: item.id ?? i,
      lat,
      lng,
      label: item.name ?? item.title ?? item.id ?? `Item ${i}`,
      properties: rest,
    };
  });
  return { markers };
}

function getCenter(mapData) {
  if (mapData.center?.lat != null) return [mapData.center.lat, mapData.center.lng];
  if (mapData.bounds) {
    return [
      (mapData.bounds.minLat + mapData.bounds.maxLat) / 2,
      (mapData.bounds.minLng + mapData.bounds.maxLng) / 2,
    ];
  }
  if (mapData.markers?.length) {
    const lats = mapData.markers.map(m => m.lat);
    const lngs = mapData.markers.map(m => m.lng);
    return [
      (Math.min(...lats) + Math.max(...lats)) / 2,
      (Math.min(...lngs) + Math.max(...lngs)) / 2,
    ];
  }
  return [0, 0];
}

function normalizeColorLegend(mapData) {
  // Support array-style legend or object-style colorLegend keyed by category
  if (Array.isArray(mapData.legend)) return mapData.legend;
  if (mapData.colorLegend && typeof mapData.colorLegend === 'object') {
    return Object.entries(mapData.colorLegend).map(([category, v]) => ({ category, ...v }));
  }
  return null;
}

function getMarkerColor(marker, legend) {
  // Prefer direct color on marker, then look up via legend
  if (marker.color) return marker.color;
  if (!legend) return '#3388ff';
  const item = legend.find(l => l.category === marker.category);
  return item?.color ?? '#3388ff';
}

function MarkerPopup({ marker }) {
  // If raw GeoJSON properties exist, render them all; otherwise use label/tooltip.
  if (marker.properties) {
    const { name, url, ...rest } = marker.properties;
    return (
      <div style={{ fontSize: 13, lineHeight: 1.5 }}>
        {name && <strong>{name}</strong>}
        {Object.entries(rest).map(([k, v]) =>
          v != null && v !== '' ? (
            <div key={k}><span style={{ color: '#666' }}>{k}:</span> {String(v)}</div>
          ) : null
        )}
        {url && (
          <div style={{ marginTop: 4 }}>
            <a href={url} target="_blank" rel="noopener noreferrer">View listing</a>
          </div>
        )}
      </div>
    );
  }
  return (
    <>
      <strong>{marker.label}</strong>
      {marker.tooltip && <p style={{ margin: '4px 0 0' }}>{marker.tooltip}</p>}
    </>
  );
}

export function canDisplayAsMap(data) {
  if (!data || typeof data !== 'object') return false;
  if (isGeoJSON(data)) {
    return data.features.some(f => f.geometry?.type === 'Point');
  }
  if (data.markers?.length > 0) return true;
  if (findCoordinatesArray(data)) return true;
  return false;
}

export default function MapResultsDisplay({ mapData }) {
  // Accept GeoJSON FeatureCollection, coordinates-array format, or internal format
  let normalized = mapData;
  if (isGeoJSON(mapData)) {
    normalized = normalizeGeoJSON(mapData);
  } else {
    const coordItems = findCoordinatesArray(mapData);
    if (coordItems) normalized = normalizeCoordinatesArray(coordItems);
  }

  if (!normalized?.markers?.length) return null;

  const center = getCenter(normalized);
  const { markers, metadata } = normalized;
  const legend = normalizeColorLegend(normalized);
  const radius = metadata?.markerSize ?? 6;

  return (
    <div style={{ width: '100%' }}>
      {normalized.title && <h2 style={{ marginBottom: 4 }}>{normalized.title}</h2>}
      {normalized.description && <p style={{ marginTop: 0, marginBottom: 8 }}>{normalized.description}</p>}

      <MapContainer
        center={center}
        zoom={7}
        style={{ height: '500px', width: '100%', borderRadius: 8 }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {markers.map(marker => (
          <CircleMarker
            key={marker.id}
            center={[marker.lat, marker.lng]}
            radius={radius}
            pathOptions={{
              color: getMarkerColor(marker, legend),
              fillColor: getMarkerColor(marker, legend),
              fillOpacity: 0.8,
              weight: 1,
            }}
          >
            <Popup><MarkerPopup marker={marker} /></Popup>
          </CircleMarker>
        ))}
      </MapContainer>

      {legend?.length > 0 && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
          {legend.map(item => (
            <span key={item.category} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13 }}>
              <span style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: item.color, display: 'inline-block', flexShrink: 0 }} />
              {item.label}{item.count != null ? ` (${item.count})` : ''}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
