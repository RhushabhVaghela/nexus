import React from 'react';
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface NexusDialogueProps {
  text: string;
  speaker?: string;
  style?: 'visual-novel' | 'subtitle' | 'cinematic';
}

export const NexusDialogue: React.FC<NexusDialogueProps> = ({
  text,
  speaker,
  style = 'visual-novel'
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opacity = spring({ frame, fps });

  if (style === 'subtitle') {
    return (
      <div style={{
        position: 'absolute',
        bottom: 50,
        width: '100%',
        textAlign: 'center',
        opacity
      }}>
        <span style={{
          backgroundColor: 'rgba(0,0,0,0.6)',
          color: 'white',
          padding: '10px 20px',
          fontSize: 40,
          borderRadius: 8,
          fontFamily: 'Arial, sans-serif'
        }}>
          {text}
        </span>
      </div>
    );
  }

  // Default Visual Novel Style
  return (
    <div style={{
      position: 'absolute',
      bottom: 20,
      left: '5%',
      width: '90%',
      height: 200,
      backgroundColor: 'rgba(0, 11, 30, 0.9)',
      border: '2px solid #00f2ff',
      borderRadius: 16,
      padding: 30,
      display: 'flex',
      flexDirection: 'column',
      opacity
    }}>
      {speaker && (
        <div style={{
          color: '#00f2ff',
          fontSize: 32,
          fontWeight: 'bold',
          marginBottom: 10
        }}>
          {speaker}
        </div>
      )}
      <div style={{
        color: 'white',
        fontSize: 36,
        fontFamily: 'Georgia, serif',
        lineHeight: 1.4
      }}>
        {text}
      </div>
    </div>
  );
};