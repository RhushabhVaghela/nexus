import React from 'react';
import { spring, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';

interface TimelineEvent {
  year: string;
  title: string;
  description?: string;
  color?: string;
}

interface NexusTimelineProps {
  events: TimelineEvent[];
  orientation?: 'horizontal' | 'vertical';
  color?: string;
}

export const NexusTimeline: React.FC<NexusTimelineProps> = ({
  events,
  orientation = 'horizontal',
  color = '#ffffff'
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // Scroll progress based on video duration
  const progress = interpolate(frame, [0, 150], [0, 1], { extrapolateRight: 'clamp' });
  
  // Calculate spacing
  const spacing = orientation === 'horizontal' ? width / 3 : height / 3;
  const totalLength = spacing * events.length;

  return (
    <div style={{ 
      width: '100%', 
      height: '100%', 
      display: 'flex', 
      justifyContent: 'center',
      alignItems: 'center',
      position: 'relative'
    }}>
      {/* Main Line */}
      <div style={{
        position: 'absolute',
        backgroundColor: color,
        width: orientation === 'horizontal' ? '200%' : 4,
        height: orientation === 'horizontal' ? 4 : '200%',
        transform: orientation === 'horizontal' 
          ? `translateX(${-progress * totalLength}px)` 
          : `translateY(${-progress * totalLength}px)`,
        transition: 'transform 0.1s linear'
      }}>
        {/* Events */}
        {events.map((evt, i) => {
          const pos = i * spacing + spacing / 2;
          const evtProgress = spring({ frame: frame - i * 30, fps });
          
          return (
            <div key={i} style={{
              position: 'absolute',
              left: orientation === 'horizontal' ? pos : -20,
              top: orientation === 'horizontal' ? -20 : pos,
              opacity: evtProgress,
              transform: `scale(${evtProgress})`
            }}>
              {/* Node */}
              <div style={{
                width: 44, height: 44, borderRadius: '50%', backgroundColor: '#000', border: `4px solid ${evt.color || color}`
              }} />
              
              {/* Text */}
              <div style={{
                position: 'absolute',
                top: orientation === 'horizontal' ? 60 : 0,
                left: orientation === 'horizontal' ? 0 : 60,
                width: 300,
                color: '#fff'
              }}>
                <h2 style={{ margin: 0, color: evt.color || color, fontSize: 32 }}>{evt.year}</h2>
                <h3 style={{ margin: '8px 0', fontSize: 24 }}>{evt.title}</h3>
                {evt.description && <p style={{ fontSize: 18, opacity: 0.8 }}>{evt.description}</p>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};