import React from 'react';
import { Img, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface NexusCharacterProps {
  src: string;
  name: string;
  position?: 'left' | 'center' | 'right';
  animation?: 'fade' | 'slide-up' | 'pop';
  scale?: number;
}

export const NexusCharacter: React.FC<NexusCharacterProps> = ({
  src,
  name,
  position = 'center',
  animation = 'fade',
  scale = 1
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const spr = spring({ frame, fps, config: { damping: 15 } });

  const getPositionStyle = (): React.CSSProperties => {
    const base = { position: 'absolute' as const, bottom: 0, height: '80%' };
    switch (position) {
      case 'left': return { ...base, left: '10%' };
      case 'right': return { ...base, right: '10%' };
      default: return { ...base, left: '50%', transform: `translateX(-50%) scale(${scale})` };
    }
  };

  const getAnimStyle = () => {
    if (animation === 'slide-up') return { transform: `translateY(${100 * (1 - spr)}%)` };
    if (animation === 'pop') return { transform: `scale(${spr})` };
    return { opacity: spr };
  };

  return (
    <div style={{ ...getPositionStyle(), ...getAnimStyle() }}>
      <Img 
        src={src} 
        style={{ 
          height: '100%', 
          objectFit: 'contain', 
          filter: 'drop-shadow(0 0 10px rgba(0,0,0,0.5))' 
        }} 
      />
    </div>
  );
};