import React from 'react';
import { Audio, staticFile } from 'remotion';

interface NexusAudioProps {
  src: string; // URL or local path in public/
  volume?: number;
  startFrame?: number;
  placeholder?: boolean;
}

export const NexusAudio: React.FC<NexusAudioProps> = ({
  src,
  volume = 1,
  startFrame = 0,
  placeholder = false
}) => {
  // If it's a placeholder, we don't render actual audio to avoid errors
  if (placeholder) return null;

  const audioSrc = src.startsWith('http') ? src : staticFile(src);

  return (
    <Audio
      src={audioSrc}
      volume={volume}
      startFrom={startFrame}
    />
  );
};