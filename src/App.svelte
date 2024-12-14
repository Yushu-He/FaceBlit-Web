<script lang="ts">
  import { onMount } from 'svelte';
  import { fade } from 'svelte/transition';
  import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
  import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
  import Styles from './lib/Styles.svelte';
  import { MP2Dlib } from './lib/MP2Dlib.ts';

  const WIDTH = 480;
  const HEIGHT = 640;

  let loading = true;

  let videoElement: HTMLVideoElement;
  let canvasElement: HTMLCanvasElement;
  let faceLandmarker: FaceLandmarker;
  let window = {
    cropParams: {
      sx: 0,
      sy: 0,
      sWidth: 0,
      sHeight: 0,
    },
  }

  let selectedStyleName = '';
  let lastLoadName = '';
  let selectedStyleData: {
    resizedImage: ImageData,
    landmarksArray: Int32Array,
    lookUpCube: Uint16Array
  } = {
    resizedImage: new ImageData(1, 1),
    landmarksArray: new Int32Array(),
    lookUpCube: new Uint16Array()
  };

  const mp2dlib = new MP2Dlib();
  let apiLoaded = false;
  let api = {
    create_image_buffer: (): number => {return 0},
    destroy_image_buffer: (p: number) => {},
    create_landmarks_buffer: (): number => {return 0},
    destroy_landmarks_buffer: (p: number) => {},
    create_lookUpCube_buffer: (): number => {return 0},
    destroy_lookUpCube_buffer: (p: number) => {},
    load_style: (styleImagep: number, styleLandmarksp: number, lookUpCubep: number): number => {return 0},
    stylizeImage: (targetImagep: number, targetLandmarksp: number, NNF_patchsize: number) => {},
  };

  let wasmModule: any;

  let styleP = {
    styleImagep: 0,
    styleLandmarksp: 0,
    lookUpCubep: 0,
  }

  let targetP = {
    targetImagep: 0,
    targetLandmarksp: 0,
  }

  function handleStylesLoaded(allFinished: boolean) {
    loading = !allFinished;
  }

  function handleStyleSelected(styleName: string, styleData: any) {
    selectedStyleName = styleName;
    selectedStyleData = styleData;
    console.log('Selected style: ', selectedStyleName);
  }

  onMount(async () => {
    // Load wasm module
    try {
      const Module = await import('./lib/face_blit.js');

      wasmModule = await Module.default({
        locateFile: (path) => {
          if (path.endsWith('.wasm')) {
            return './wasm/face_blit.wasm';
          }
          return path;
        },
      });

      api = {
        create_image_buffer: wasmModule.cwrap('create_image_buffer', 'number', []),
        destroy_image_buffer: wasmModule.cwrap('destroy_image_buffer', '', ['number']),
        create_landmarks_buffer: wasmModule.cwrap('create_landmarks_buffer', 'number', []),
        destroy_landmarks_buffer: wasmModule.cwrap('destroy_landmarks_buffer', '', ['number']),
        create_lookUpCube_buffer: wasmModule.cwrap('create_lookUpCube_buffer', 'number', []),
        destroy_lookUpCube_buffer: wasmModule.cwrap('destroy_lookUpCube_buffer', '', ['number']),
        load_style: wasmModule.cwrap('load_style', 'number', ['number', 'number', 'number']),
        stylizeImage: wasmModule.cwrap('stylizeImage', '', ['number', 'number', 'number']),
      };
      apiLoaded = true;
    } catch (e) {
      console.error('Error loading face_blit.js:', e);
    }

    // create style buffers
    styleP.styleImagep = api.create_image_buffer();
    styleP.styleLandmarksp = api.create_landmarks_buffer(); 
    styleP.lookUpCubep = api.create_lookUpCube_buffer();

    // create target buffers
    targetP.targetImagep = api.create_image_buffer();
    targetP.targetLandmarksp = api.create_landmarks_buffer();

    // Load face landmark model
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: './assets/models/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: false,
      runningMode: 'VIDEO',
      numFaces: 1,
    });

    // Start camera
    startCamera();
  });

  function transformLandmarksToInt32Array(landmarks: NormalizedLandmark[]): Int32Array {
    const landmarksArray = new Int32Array(landmarks.length * 2);
    for (let i = 0; i < landmarks.length; i++) {
      landmarksArray[i * 2] = Math.round(landmarks[i].x * WIDTH);
      landmarksArray[i * 2 + 1] = Math.round(landmarks[i].y * HEIGHT);
      if (landmarksArray[i * 2] < 0 || landmarksArray[i * 2] >= WIDTH
        || landmarksArray[i * 2 + 1] < 0 || landmarksArray[i * 2 + 1] >= HEIGHT
      ) {
        return null;
      }
    }
    return landmarksArray;
  }

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
      height: { ideal: 720 },
      facingMode: 'user'
      }
    });
    videoElement.srcObject = stream;
    videoElement.play();

    videoElement.onloadedmetadata = () => {
      // get original video resolution
      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;
      // calculate resolution ratio crop parameters
      const videoAspectRatio = videoWidth / videoHeight;
      const desiredAspectRatio = WIDTH / HEIGHT;

      if (videoAspectRatio > desiredAspectRatio) {
        window.cropParams.sHeight = videoHeight;
        window.cropParams.sWidth = videoHeight * desiredAspectRatio;
        window.cropParams.sx = (videoWidth - window.cropParams.sWidth) / 2;
        window.cropParams.sy = 0;
      } else {
        window.cropParams.sWidth = videoWidth;
        window.cropParams.sHeight = videoWidth / desiredAspectRatio;
        window.cropParams.sx = 0;
        window.cropParams.sy = (videoHeight - window.cropParams.sHeight) / 2;
      }

      canvasElement.width = WIDTH;
      canvasElement.height = HEIGHT;

      console.log(window.cropParams);

      detectFaces();
    };

  function detectFaces() {
    const canvasCtx = canvasElement.getContext('2d', {willReadFrequently: true});
    const drawingUtils = new DrawingUtils(canvasCtx);
    const { sx, sy, sWidth, sHeight } = window.cropParams;

    async function render() {
      canvasCtx.drawImage(
        videoElement,
        sx, sy, sWidth, sHeight,
        0, 0, canvasElement.width, canvasElement.height 
      );
      const results = faceLandmarker.detectForVideo(canvasElement, Date.now());

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks_extracted = mp2dlib.transformLandmarks(results.faceLandmarks);
        const targetLandmarksArray = transformLandmarksToInt32Array(landmarks_extracted[0]);
        if (selectedStyleName && selectedStyleData!.landmarksArray!.length > 0 && targetLandmarksArray) {
          if (selectedStyleName !== lastLoadName) {
            lastLoadName = selectedStyleName;
            wasmModule.HEAPU8.set(selectedStyleData.resizedImage.data, styleP.styleImagep);
            wasmModule.HEAPU16.set(selectedStyleData.lookUpCube, styleP.lookUpCubep / 2);
            wasmModule.HEAP32.set(selectedStyleData.landmarksArray, styleP.styleLandmarksp / 4);
            const styleLoadWrong = await api.load_style(styleP.styleImagep, styleP.styleLandmarksp, styleP.lookUpCubep);
            if (styleLoadWrong) {
              console.error('Error loading style:', selectedStyleName);
            }
          }
          // canvas image data
          const targetImageData = canvasCtx.getImageData(0, 0, WIDTH, HEIGHT);
          wasmModule.HEAPU8.set(targetImageData.data, targetP.targetImagep);
          wasmModule.HEAP32.set(targetLandmarksArray, targetP.targetLandmarksp / 4);
          await api.stylizeImage(targetP.targetImagep, targetP.targetLandmarksp, 0);

          // set target image pointer data to canvas
          const targetImageDataArray = new Uint8ClampedArray(wasmModule.HEAPU8.buffer, targetP.targetImagep, WIDTH * HEIGHT * 4);
          const ReturntargetImageData = new ImageData(targetImageDataArray, WIDTH, HEIGHT);
          // clean canvas
          canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);
          // draw new image
          canvasCtx.putImageData(ReturntargetImageData, 0, 0);
        }
        else {
          drawingUtils.drawLandmarks(landmarks_extracted[0], { color: '#FF3030' , radius: 0.5 });
        }
        // for (const landmarks of landmarks_extracted) {
        //   drawingUtils.drawConnectors(
        //     landmarks,
        //     FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        //     { color: '#C0C0C070', lineWidth: 1 }
        //   );
        //   drawingUtils.drawConnectors(
        //     landmarks,
        //     FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        //     { color: '#30FF30' }
        //   );
        //   drawingUtils.drawConnectors(
        //     landmarks,
        //     FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        //     { color: '#FF3030' }
        //   );
      }

      requestAnimationFrame(render);
    }

    render();
  }
}
</script>


<main>
  <div class="loading-page" class:hidden={!loading} transition:fade={{ duration: 1000 }}>
    <div class="loading-circle"></div>
    <h1>Loading...</h1>
  </div>
  <div class="app page" class:hidden={loading} transition:fade={{ duration: 1000 }}>
    <h1>EECS 442 Project: Face Blit Web</h1>
    <p><a href="https://github.com/Yushu-He/FaceBlit-Web" target="_blank" class="github-link" style="text-decoration: none; color: inherit;">
      <i class="fa fa-github" style="font-size: 24px; margin-right: 8px;"></i> View on GitHub
    </a></p>
    <video bind:this={videoElement} playsinline autoplay muted style="display: none;" aria-hidden="true"></video>
    <canvas bind:this={canvasElement}></canvas>
    <h1>Styles</h1>
    <Styles 
      stylesLoaded={handleStylesLoaded}
      styleSelected={handleStyleSelected}
    />
  </div>
</main>

<style>
  canvas {
    width: auto;
    height: 100%;
    border-radius: 60px;
    transform: scaleX(-1);
  }
  .loading-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }
  .loading-circle {
    width: 50px;
    height: 50px;
    border: 5px solid #ccc;
    border-top: 5px solid #333;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 20px;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .hidden {
    display: none;
  }
</style>