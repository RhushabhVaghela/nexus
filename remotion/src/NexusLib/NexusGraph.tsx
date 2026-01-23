import React from 'react';
import { spring, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';

interface NexusGraphProps {
  fn: (x: number) => number;
  xRange?: [number, number];
  yRange?: [number, number];
  width?: number;
  height?: number;
  color?: string;
  strokeWidth?: number;
}

export const NexusGraph: React.FC<NexusGraphProps> = ({
  fn,
  xRange = [-5, 5],
  yRange = [-5, 5],
  width = 800,
  height = 600,
  color = '#00f2ff',
  strokeWidth = 3
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame,
    fps,
    config: { stiffness: 40 }
  });

  const points: [number, number][] = [];
  const resolution = 100;
  
  for (let i = 0; i <= resolution; i++) {
    const x = xRange[0] + (xRange[1] - xRange[0]) * (i / resolution);
    const y = fn(x);
    points.push([x, y]);
  }

  const mapX = (x: number) => (x - xRange[0]) / (xRange[1] - xRange[0]) * width;
  const mapY = (y: number) => height - (y - yRange[0]) / (yRange[1] - yRange[0]) * height;

  const pathData = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p[0])} ${mapY(p[1])}`)
    .join(' ');

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Axes */}
      <line x1={0} y1={mapY(0)} x2={width} y2={mapY(0)} stroke="#666" strokeWidth={1} />
      <line x1={mapX(0)} y1={0} x2={mapX(0)} y2={height} stroke="#666" strokeWidth={1} />
      
      {/* Function Path */}
      <path
        d={pathData}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={10000}
        strokeDashoffset={10000 * (1 - progress)}
      />
    </svg>
  );
};