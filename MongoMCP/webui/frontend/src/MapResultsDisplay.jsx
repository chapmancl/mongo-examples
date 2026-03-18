import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

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

function getCategoryColor(category, legend) {
  if (!legend) return '#3388ff';
  const item = legend.find(l => l.category === category);
  return item?.color ?? '#3388ff';
}

export default function MapResultsDisplay({ mapData }) {
  if (!mapData?.markers?.length) return null;

  const center = getCenter(mapData);
  const { markers, legend, metadata } = mapData;
  const radius = metadata?.markerSize ?? 6;

  return (
    <div style={{ width: '100%' }}>
      {mapData.title && <h2 style={{ marginBottom: 4 }}>{mapData.title}</h2>}
      {mapData.description && <p style={{ marginTop: 0, marginBottom: 8 }}>{mapData.description}</p>}

      <MapContainer
        center={center}
        zoom={6}
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
              color: getCategoryColor(marker.category, legend),
              fillColor: getCategoryColor(marker.category, legend),
              fillOpacity: 0.8,
              weight: 1,
            }}
          >
            <Popup>
              <strong>{marker.label}</strong>
              {marker.tooltip && <p style={{ margin: '4px 0 0' }}>{marker.tooltip}</p>}
            </Popup>
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
