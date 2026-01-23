import React from 'react';
import { spring, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface NexusMathProps {
  latex: string;
  block?: boolean;
  animate?: 'fade' | 'write' | 'none';
  fontSize?: number;
  color?: string;
}

export const NexusMath: React.FC<NexusMathProps> = ({
  latex,
  block = true,
  animate = 'write',
  fontSize = 40,
  color = '#fff'
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const spr = spring({
    frame,
    fps,
    config: {
      damping: 12,
    },
  });

  const opacity = interpolate(spr, [0, 1], [0, 1]);
  const scale = interpolate(spr, [0, 1], [0.8, 1]);

  const style: React.CSSProperties = {
    fontSize: `${fontSize}px`,
    color,
    opacity: animate !== 'none' ? opacity : 1,
    transform: animate !== 'none' ? `scale(${scale})` : 'none',
    fontFamily: 'KaTeX_Main, Times New Roman, serif',
  };

  return (
    <div style={style}>
      {block ? <BlockMath math={latex} /> : <InlineMath math={latex} />}
    </div>
  );
};