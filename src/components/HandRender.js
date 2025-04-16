import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

const HandRender = () => {
  
  const rendererContainer = useRef(null);
  const [loadedMeshHand, setLoadedMeshHand] = useState(null);
  const [loadedMeshMouse, setLoadedMeshMouse] = useState(null);

  useEffect(() => {

      if (rendererContainer.current) {
    while (rendererContainer.current.firstChild) {
      rendererContainer.current.removeChild(rendererContainer.current.firstChild);
    }
  }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const canvasWidth = 280;
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
    spotLightFront.intensity = 5;
    spotLightFront.distance = 50;
    spotLightFront.decay = 1;
    scene.add(spotLightFront);

    // Load environment map
    const cubeTextureLoader = new THREE.CubeTextureLoader();
    const envMap = cubeTextureLoader.load([]);

    const modelGroup = new THREE.Group();
    scene.add(modelGroup);

    // Load Hand Model
    new GLTFLoader().load(
      '/hand/scene.gltf',
      (gltf) => {
        console.log('loading hand model');
        const handMesh = gltf.scene;

        const material = new THREE.MeshStandardMaterial({
          metalness: 1,
          roughness: 0.0,
          envMap: envMap,
        });

        handMesh.traverse((child) => {
          if (child.isMesh) {
            child.material = material;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        handMesh.position.set(0, 0, -1.5);
        handMesh.rotation.set(0.0, 0, 0);
        handMesh.scale.set(1.2, 1.2, 1.2);
        setLoadedMeshHand(handMesh);
        modelGroup.add(handMesh); // Add the hand to the group
      },
      (xhr) => {
        console.log(`loading ${Math.round((xhr.loaded / xhr.total) * 100)}%`);
      },
      (error) => {
        console.error(error);
      }
    );

    // Load Mouse Model
    new GLTFLoader().load(
      '/mouse/scene.gltf',
      (gltf) => {
        console.log('loading mouse model');
        const mouseMesh = gltf.scene;

        const mouseMaterial = new THREE.MeshStandardMaterial({
          metalness: 1,
          roughness: 0.3,
          envMap: envMap,
        });

        new RGBELoader().load('pixelcut-export.hdr', (environmentMap) => {
          environmentMap.mapping = THREE.EquirectangularReflectionMapping;

          scene.environment = environmentMap;

          if (loadedMeshHand) {
            loadedMeshHand.traverse((child) => {
                if (child.isMesh) {
                    child.material.envMap = environmentMap;
                }
            });
        }
        });

        mouseMesh.traverse((child) => {
          if (child.isMesh) {
            child.material = mouseMaterial;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        mouseMesh.position.set(0, 0.6, -0.8);
        mouseMesh.rotation.set(-Math.PI / 2 - 0.2, Math.PI / 2, 0);
        mouseMesh.scale.set(0.75, 0.75, 0.75);

        setLoadedMeshMouse(mouseMesh);
        modelGroup.add(mouseMesh); // Add the mouse to the group
      },
      (xhr) => {
        console.log(`loading mouse model ${Math.round((xhr.loaded / xhr.total) * 100)}%`);
      },
      (error) => {
        console.error(error);
      }
    );

    // Mouse movement
    const mouse = new THREE.Vector2();
    const onMouseMove = (event) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1; // Normalized X
      mouse.y = (event.clientY / window.innerHeight) * 2 - 0.75; // Normalized Y
    };

    window.addEventListener('mousemove', onMouseMove);

    const animate = () => {
      requestAnimationFrame(animate);

      if (modelGroup) {
        modelGroup.rotation.y = mouse.x * Math.PI * 0.1 + Math.PI; // Inverted Y rotation
        modelGroup.rotation.x = mouse.y * Math.PI * 0.1;
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
    };
  }, []);

  return <div ref={rendererContainer} className="renderer-container"  />;
};

export default HandRender;
