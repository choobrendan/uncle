import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

const FaceRender = ({X,Y}) => {
  const rendererContainer = useRef(null);
  const [loadedMesh, setLoadedMesh] = useState(null);

  useEffect(() => {
    if (rendererContainer.current) {
      while (rendererContainer.current.firstChild) {
        rendererContainer.current.removeChild(rendererContainer.current.firstChild);
      }
    }
    let mesh = null;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const canvasWidth = 320;
    const canvasHeight = window.innerHeight;
    renderer.setSize(canvasWidth, canvasHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    if (rendererContainer.current) {
      rendererContainer.current.appendChild(renderer.domElement);
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, canvasWidth / canvasHeight, 1, 1000);
    camera.position.set(0, 1, 8);

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff));

    const addSpotlight = (x, y, z, intensity, color = 0xffffff) => {
      const light = new THREE.SpotLight(color);
      light.position.set(x, y, z);
      light.castShadow = true;
      light.angle = Math.PI / 4;
      light.penumbra = 0.5;
      light.intensity = intensity;
      light.distance = 50;
      light.decay = 1;
      scene.add(light);
    };

    addSpotlight(0, 20, 0, 30, 0xd7ffbe);
    addSpotlight(0, -20, 0, 30, 0xd7ffbe);
    addSpotlight(0, 0, 10, 20);

    // Load Model
    new GLTFLoader().load(
      '/head/scene.gltf',
      (gltf) => {
        mesh = gltf.scene;
        const material = new THREE.MeshStandardMaterial({
          metalness: 1,
          roughness: 0.0,
          color: '#ffffff',
        });

        new RGBELoader().load('pixelcut-export.hdr', (envMap) => {
          envMap.mapping = THREE.EquirectangularReflectionMapping;
          scene.environment = envMap;

          mesh.traverse((child) => {
            if (child.isMesh) {
              child.material.envMap = envMap;
            }
          });
        });

        mesh.traverse((child) => {
          if (child.isMesh) {
            child.material = material;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        mesh.position.set(0, 1.05, 1.5);
        scene.add(mesh);
        setLoadedMesh(mesh);
      },
      undefined,
      (error) => {
        console.error('Error loading model:', error);
      }
    );
const mouse = new THREE.Vector2();

const onMouseMove = (event) => {
  console.log(X)
  mouse.x = (X / window.innerWidth) * 2 - 1; // Normalized X
  mouse.y = (Y/ window.innerHeight) * 2 - 0.75; // Normalized Y
};

    window.addEventListener('mousemove', onMouseMove);

    const animate = () => {
      requestAnimationFrame(animate);

      if (mesh) {
        // Smooth and intuitive rotation based on mouse
        mesh.rotation.y  = mouse.x * Math.PI * 0.1 ; // Inverted Y rotation
        mesh.rotation.x = mouse.y * Math.PI * 0.1;
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
    };
  }, []);

  return <div ref={rendererContainer} className="renderer-container" />;
};

export default FaceRender;
