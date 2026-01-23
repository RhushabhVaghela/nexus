import React from 'react';
import { spring, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
}

interface Edge {
  from: string;
  to: string;
}

interface NexusFlowProps {
  nodes: Node[];
  edges: Edge[];
  color?: string;
}

export const NexusFlow: React.FC<NexusFlowProps> = ({
  nodes,
  edges,
  color = '#00ff88'
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame,
    fps,
    config: { stiffness: 60 }
  });

  const nodeRadius = 60;

  return (
    <svg width="100%" height="100%" viewBox="0 0 1920 1080">
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="10"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={color} />
        </marker>
      </defs>

      {/* Render Edges */}
      {edges.map((edge, i) => {
        const fromNode = nodes.find(n => n.id === edge.from);
        const toNode = nodes.find(n => n.id === edge.to);
        if (!fromNode || !toNode) return null;

        return (
          <line
            key={`edge-${i}`}
            x1={fromNode.x}
            y1={fromNode.y}
            x2={interpolate(progress, [0, 1], [fromNode.x, toNode.x])}
            y2={interpolate(progress, [0, 1], [fromNode.y, toNode.y])}
            stroke={color}
            strokeWidth={4}
            markerEnd="url(#arrowhead)"
            opacity={progress}
          />
        );
      })}

      {/* Render Nodes */}
      {nodes.map((node) => {
        const nodeProgress = spring({
          frame: frame - 10,
          fps,
          config: { damping: 12 }
        });

        return (
          <g key={node.id} transform={`scale(${nodeProgress})`} style={{ transformOrigin: `${node.x}px ${node.y}px` }}>
            <circle
              cx={node.x}
              cy={node.y}
              r={nodeRadius}
              fill="#111"
              stroke={color}
              strokeWidth={3}
            />
            <text
              x={node.x}
              y={node.y}
              textAnchor="middle"
              alignmentBaseline="middle"
              fill="#fff"
              fontSize={24}
              fontFamily="Arial"
            >
              {node.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};