<script lang="ts">
  import { onMount } from 'svelte';
  import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
  import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
  import Styles from './lib/Styles.svelte';
  import { MP2Dlib } from './lib/MP2Dlib.ts';

  const CUBE_SIZE = 256;
  const WIDTH = 768;
  const HEIGHT = 1024;
  const LANDMARKS_SIZE = 68;

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
  let loeaded = false;
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

  function handleStyleSelected(styleName: string, styleData: any) {
    selectedStyleName = styleName;
    selectedStyleData = styleData;
    console.log('Selected style: ', selectedStyleName);
  }

  onMount(async () => {
    // Load wasm module
    try {
      const Module = await import('../public/wasm/face_blit.js');

      wasmModule = await Module.default({
        locateFile: (path) => {
          if (path.endsWith('.wasm')) {
            return '/wasm/face_blit.wasm';
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
      loeaded = true;
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
        modelAssetPath: '/assets/models/face_landmarker.task',
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
      landmarksArray[i * 2] = Math.round(landmarks[i].x * 768);
      landmarksArray[i * 2 + 1] = Math.round(landmarks[i].y * 1024);
    }
    return landmarksArray;
  }

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
      width: { ideal: 1920 },
      height: { ideal: 1080 },
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
        if (selectedStyleName && selectedStyleData!.landmarksArray!.length > 0) {
          
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
  <header>
    <h1>EECS 442: Computer Vision Project</h1>
    <p>
      This project demonstrates real-time face stylization using WebAssembly and MediaPipe Vision Tasks.
    </p>
  </header>
  <section class="content">
    <div class="canvas-container">
      <video bind:this="{videoElement}" style="display: none;" aria-hidden="true"></video>
      <canvas bind:this="{canvasElement}"></canvas>
    </div>
    <Styles onStyleSelected={handleStyleSelected} cropParams={window.cropParams}/>
  </section>
  <footer>
    <p>Project by Yushu He, Ruiqi He, Wentao Wei, Shihui Sun | EECS 442, Winter 2024</p>
  </footer>
</main>

<style>
  /* General Styles */
  main {
    font-family: Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    height: 100vh;
    padding: 0;
    margin: 0;
    background: linear-gradient(to bottom, #1e3c72, #2a5298); /* Fancy gradient background */
    color: #fff;
    text-align: center;
    width: 100%;
  }

  header {
    padding: 10px 20px;
  }

  h1 {
    font-size: 2rem;
    margin: 0;
    color: #ffd700; /* Gold color for title */
  }

  p {
    font-size: 1rem;
    margin-top: 5px;
    max-width: 600px;
    line-height: 1.5;
  }

  .content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 20px;
  }

  .canvas-container {
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.1); /* Semi-transparent background for canvas */
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  canvas {
    border: 2px solid #ffffff;
    border-radius: 8px;
    max-width: 90%;
    max-height: 80vh; /* Ensures canvas fits within the viewport */
  }

  footer {
    padding: 10px;
    font-size: 0.9rem;
    color: #ddd;
  }
</style>