import React from 'react';
import { Img, staticFile } from 'remotion';

interface Marker {
  lat: number;
  lng: number;
  label: string;
}

interface NexusMapProps {
  markers?: Marker[];
  highlightCountry?: string;
  style?: 'dark' | 'satellite';
}

// Simplification: In a real prod env, we'd use 'react-simple-maps' or Leaflet images
// For this generator, we'll use a high-quality abstract world map asset
export const NexusMap: React.FC<NexusMapProps> = ({ markers = [] }) => {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', backgroundColor: '#000b1e' }}>
      {/* Placeholder World Map SVG */}
      <svg viewBox="0 0 1000 500" style={{ width: '100%', height: '100%', opacity: 0.3 }}>
         <rect width="1000" height="500" fill="#00152a" />
         <text x="500" y="250" fill="#004466" fontSize="50" textAnchor="middle">WORLD MAP VISUALIZATION</text>
         {/* Real implementation would load a geojson path here */}
      </svg>
      
      {markers.map((m, i) => (
        <div key={i} style={{
          position: 'absolute',
          left: `${(m.lng + 180) * (100/360)}%`,
          top: `${(90 - m.lat) * (100/180)}%`,
          width: 20, height: 20, borderRadius: '50%', backgroundColor: '#ff0055'
        }}>
          <div style={{ position: 'absolute', top: 25, color: 'white', width: 200 }}>{m.label}</div>
        </div>
      ))}
    </div>
  );
};