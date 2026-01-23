import React, { useMemo } from 'react';
import { ThreeCanvas } from '@remotion/three';
import { useVideoConfig } from 'remotion';
import * as THREE from 'three';

interface Vector3D {
  x: number;
  y: number;
  z: number;
  color?: string;
  label?: string;
}

interface Nexus3DProps {
  vectors?: Vector3D[];
  showGrid?: boolean;
  cameraPosition?: [number, number, number];
}

const Vector: React.FC<Vector3D> = ({ x, y, z, color = '#00f2ff' }) => {
  const origin = new THREE.Vector3(0, 0, 0);
  const dir = new THREE.Vector3(x, y, z);
  const length = dir.length();
  dir.normalize();

  return (
    <primitive
      object={new THREE.ArrowHelper(dir, origin, length, color, 0.2, 0.1)}
    />
  );
};

export const Nexus3D: React.FC<Nexus3DProps> = ({
  vectors = [],
  showGrid = true,
  cameraPosition = [5, 5, 5]
}) => {
  const { width, height } = useVideoConfig();

  return (
    <ThreeCanvas width={width} height={height}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <perspectiveCamera
        makeDefault
        position={cameraPosition}
        fov={75}
      />
      
      {showGrid && (
        <primitive object={new THREE.GridHelper(10, 10, 0x444444, 0x222222)} />
      )}
      
      <primitive object={new THREE.AxesHelper(5)} />

      {vectors.map((v, i) => (
        <Vector key={`v-${i}`} {...v} />
      ))}
    </ThreeCanvas>
  );
};