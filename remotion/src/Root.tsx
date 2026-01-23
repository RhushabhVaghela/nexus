import React from 'react';
import { Composition } from 'remotion';
import { NexusMath } from './NexusLib/NexusMath';
import { NexusGraph } from './NexusLib/NexusGraph';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MathExample"
        component={NexusMath}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          latex: 'e^{i\\\\pi} + 1 = 0',
          fontSize: 80,
          color: '#00f2ff'
        }}
      />
      <Composition
        id="GraphExample"
        component={() => (
           <div style={{ flex: 1, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
             <NexusGraph fn={(x) => Math.sin(x)} />
           </div>
        )}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};