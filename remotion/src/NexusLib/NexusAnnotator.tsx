import React from 'react';
import { spring, useCurrentFrame, useVideoConfig, Img, interpolate } from 'remotion';

interface Annotation {
  x: number;
  y: number;
  text: string;
  delay?: number;
}

interface NexusAnnotatorProps {
  src: string;
  annotations: Annotation[];
  pulseColor?: string;
}

export const NexusAnnotator: React.FC<NexusAnnotatorProps> = ({
  src,
  annotations,
  pulseColor = '#ff0055'
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Img
        src={src}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain'
        }}
      />
      
      {annotations.map((ann, i) => {
        const startFrame = ann.delay || i * 20;
        const progress = spring({
          frame: frame - startFrame,
          fps,
          config: { damping: 10 }
        });

        const pulseScale = interpolate(
          (frame - startFrame) % 30,
          [0, 30],
          [1, 2]
        );
        const pulseOpacity = interpolate(
          (frame - startFrame) % 30,
          [0, 30],
          [0.6, 0]
        );

        if (frame < startFrame) return null;

        return (
          <div
            key={`ann-${i}`}
            style={{
              position: 'absolute',
              left: `${ann.x}%`,
              top: `${ann.y}%`,
              transform: `scale(${progress})`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}
          >
            {/* Pulse Effect */}
            <div
              style={{
                position: 'absolute',
                width: 20,
                height: 20,
                borderRadius: '50%',
                backgroundColor: pulseColor,
                transform: `scale(${pulseScale})`,
                opacity: pulseOpacity,
              }}
            />
            
            {/* Dot */}
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                backgroundColor: pulseColor,
                border: '2px solid white'
              }}
            />
            
            {/* Label */}
            <div
              style={{
                marginTop: 8,
                padding: '4px 12px',
                backgroundColor: 'rgba(0,0,0,0.8)',
                color: 'white',
                borderRadius: 4,
                fontSize: 18,
                whiteSpace: 'nowrap',
                border: `1px solid ${pulseColor}`
              }}
            >
              {ann.text}
            </div>
          </div>
        );
      })}
    </div>
  );
};