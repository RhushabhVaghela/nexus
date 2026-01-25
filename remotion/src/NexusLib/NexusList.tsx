import React from 'react';
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface ListItem {
  title: string;
  detail?: string;
  icon?: string;
}

interface NexusListProps {
  items: ListItem[];
  type?: 'bullet' | 'number' | 'checklist';
  color?: string;
}

export const NexusList: React.FC<NexusListProps> = ({
  items,
  type = 'number',
  color = '#00f2ff'
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ width: '100%', padding: 80, display: 'flex', flexDirection: 'column', gap: 40 }}>
      {items.map((item, i) => {
        const progress = spring({ frame: frame - i * 15, fps, config: { damping: 12 } });
        
        return (
          <div key={i} style={{ 
            transform: `translateX(${(1 - progress) * -100}px)`, 
            opacity: progress, 
            display: 'flex', 
            alignItems: 'center', 
            gap: 30,
            backgroundColor: 'rgba(255,255,255,0.05)',
            padding: 20,
            borderRadius: 16,
            borderLeft: `6px solid ${color}`
          }}>
            {/* Indicator */}
            <div style={{ 
              width: 60, height: 60, borderRadius: '50%', 
              backgroundColor: color, color: '#000', 
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 32, fontWeight: 'bold'
            }}>
              {type === 'number' ? i + 1 : type === 'checklist' ? '✓' : '•'}
            </div>
            
            {/* Text */}
            <div>
              <h2 style={{ color: 'white', margin: 0, fontSize: 40 }}>{item.title}</h2>
              {item.detail && <p style={{ color: '#aaa', margin: '5px 0 0 0', fontSize: 24 }}>{item.detail}</p>}
            </div>
          </div>
        );
      })}
    </div>
  );
};