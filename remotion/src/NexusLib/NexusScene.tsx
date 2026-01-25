import React from 'react';
import { Img, AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';

interface NexusSceneProps {
  background: string;
  filter?: 'none' | 'sepia' | 'grayscale' | 'night' | 'dream';
  zoom?: boolean;
  children?: React.ReactNode;
}

export const NexusScene: React.FC<NexusSceneProps> = ({
  background,
  filter = 'none',
  zoom = false,
  children
}) => {
  const frame = useCurrentFrame();
  
  // Subtle Ken Burns zoom effect
  const scale = zoom ? interpolate(frame, [0, 300], [1, 1.1]) : 1;

  const getFilterStyle = () => {
    switch (filter) {
      case 'sepia': return 'sepia(0.8)';
      case 'grayscale': return 'grayscale(1)';
      case 'night': return 'brightness(0.4) contrast(1.2) blue(20%)';
      case 'dream': return 'blur(2px) brightness(1.2)';
      default: return 'none';
    }
  };

  return (
    <AbsoluteFill style={{ overflow: 'hidden', backgroundColor: '#000' }}>
      <Img
        src={background}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: `scale(${scale})`,
          filter: getFilterStyle(),
        }}
      />
      <AbsoluteFill>
        {children}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};