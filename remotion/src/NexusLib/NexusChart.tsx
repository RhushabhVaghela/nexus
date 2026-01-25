import React from 'react';
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface DataPoint {
  label: string;
  value: number;
  color?: string;
}

interface NexusChartProps {
  data: DataPoint[];
  type?: 'bar' | 'pie';
  title?: string;
  maxVal?: number;
}

export const NexusChart: React.FC<NexusChartProps> = ({
  data,
  type = 'bar',
  title,
  maxVal
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const maxValue = maxVal || Math.max(...data.map(d => d.value)) * 1.1;

  return (
    <div style={{ width: '100%', height: '100%', padding: 100, display: 'flex', flexDirection: 'column' }}>
      {title && <h1 style={{ color: 'white', fontSize: 60, marginBottom: 60, textAlign: 'center' }}>{title}</h1>}
      
      {/* Bar Chart Implementation */}
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        alignItems: 'flex-end', 
        justifyContent: 'space-around',
        borderBottom: '2px solid #444',
        borderLeft: '2px solid #444',
        padding: 20
      }}>
        {data.map((d, i) => {
          const progress = spring({ frame: frame - i * 10, fps });
          const barHeight = (d.value / maxValue) * 100;
          
          return (
            <div key={i} style={{ 
              height: `${barHeight * progress}%`, 
              width: `${80 / data.length}%`, 
              backgroundColor: d.color || '#00f2ff',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              position: 'relative',
              borderRadius: '8px 8px 0 0'
            }}>
              <span style={{ position: 'absolute', top: -40, color: 'white', fontSize: 24, opacity: progress }}>{d.value}</span>
              <span style={{ position: 'absolute', bottom: -50, color: 'white', fontSize: 24, opacity: progress }}>{d.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};