import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

const FaceRender = ({ X, Y }) => {
  const rendererContainer = useRef(null);
  const [loadedMesh, setLoadedMesh] = useState(null);

  const updateMeshRotation = (x, y) => {
    const normalizedX = -(x / window.innerWidth) * 2 + 1;
    const normalizedY = -(y / window.innerHeight) * 2 - 0.75;

    if (loadedMesh) {
      loadedMesh.rotation.y = -normalizedX * Math.PI * 0.2; // Inverted Y rotation
      loadedMesh.rotation.x = normalizedY * Math.PI * 0.1; // Normal X rotation
    }
  };

  useEffect(() => {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const canvasWidth = 320;
    const canvasHeight = window.innerHeight;
    renderer.setSize(canvasWidth, canvasHeight);
    renderer.setClearColor(0x000000, 0); // Set to transparent
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    if (rendererContainer.current) {
      rendererContainer.current.appendChild(renderer.domElement);
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, canvasWidth / canvasHeight, 1, 1000);
    camera.position.set(0, 1, 8);

    const light = new THREE.AmbientLight(0xffffff);
    scene.add(light);

    const spotLightTop = new THREE.SpotLight(0xd7ffbe);
    spotLightTop.position.set(0, 20, 0);
    spotLightTop.angle = Math.PI / 4;
    spotLightTop.penumbra = 2;
    spotLightTop.castShadow = true;
    spotLightTop.intensity = 30;
    spotLightTop.distance = 50;
    spotLightTop.decay = 1;
    scene.add(spotLightTop);

    const spotLightBottom = new THREE.SpotLight(0xd7ffbe);
    spotLightBottom.position.set(0, -20, 0);
    spotLightBottom.angle = Math.PI / 4;
    spotLightBottom.penumbra = 0.1;
    spotLightBottom.castShadow = true;
    spotLightBottom.intensity = 30;
    spotLightBottom.distance = 50;
    spotLightBottom.decay = 1;
    scene.add(spotLightBottom);

    const spotLightFront = new THREE.SpotLight(0xffffff);
    spotLightFront.position.set(0, 0, 10);
    spotLightFront.angle = Math.PI / 4;
    spotLightFront.penumbra = 0.1;
    spotLightFront.castShadow = true;
    spotLightFront.intensity = 20;
    spotLightFront.distance = 50;
    spotLightFront.decay = 1;
    scene.add(spotLightFront);

    new GLTFLoader().load(
      '/head/scene.gltf',
      (gltf) => {
        const mesh = gltf.scene;
        const material = new THREE.MeshStandardMaterial({
          metalness: 1,
          roughness: 0.0,
          color: '#ffffff',
        });
        new RGBELoader().load('pixelcut-export.hdr', (environmentMap) => {
          environmentMap.mapping = THREE.EquirectangularReflectionMapping;
          scene.environment = environmentMap;

          if (mesh) {
            mesh.traverse((child) => {
              if (child.isMesh) {
                child.material.envMap = environmentMap;
              }
            });
          }
        });

        mesh.traverse((child) => {
          if (child.isMesh) {
            child.material = material;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        mesh.position.set(0, 1.05, 1.5);
        setLoadedMesh(mesh); // Update the loaded mesh
        scene.add(mesh);
      },
      (xhr) => {
        console.log(`loading ${Math.round((xhr.loaded / xhr.total) * 100)}%`);
      },
      (error) => {
        console.error(error);
      }
    );

    // Track mouse movement
    const mouse = new THREE.Vector2();
    const onMouseMove = (event) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = (event.clientY / window.innerHeight) * 2 - 0.75;
    };

    window.addEventListener('mousemove', onMouseMove);

    const animate = () => {
      requestAnimationFrame(animate);

      // Rotate the model group based on mouse position
      if (loadedMesh) {
        loadedMesh.rotation.y = mouse.x * Math.PI * 0.1; // Inverted Y rotation
        loadedMesh.rotation.x = mouse.y * Math.PI * 0.1;
      }

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup on unmount
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
    };
  }, []);

  // Watch for X and Y changes to update rotation
  useEffect(() => {
    updateMeshRotation(X, Y);
  }, [X, Y]);

  return <div ref={rendererContainer} className="renderer-container"></div>;
};

export default FaceRender;
