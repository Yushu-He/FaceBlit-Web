<script lang="ts">
  import { onMount } from 'svelte';
  import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
  import Styles from './lib/Styles.svelte';
  import { MP2Dlib } from './lib/MP2Dlib.ts';
  import { StyleTransfer } from './lib/StyleTransfer.ts';

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
    globalScale: 1,
  }

  let selectedStyleName = '';
  let selectedStyleData = {};

  const mp2dlib = new MP2Dlib();
  const styleTransfer = new StyleTransfer();

  function handleStyleSelected(styleName: string, styleData: any) {
    selectedStyleName = styleName;
    selectedStyleData = styleData;
    console.log('Selected style: ', selectedStyleName);
  }

  onMount(async () => {
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

    startCamera();
  });

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    videoElement.play();

    videoElement.onloadedmetadata = () => {
      // set target resolution
      const desiredWidth = 768;
      const desiredHeight = 1024;

      // get original video resolution
      const videoWidth = videoElement.videoWidth;
      const videoHeight = videoElement.videoHeight;
      let sx: number, sy: number, sWidth: number, sHeight: number;
      // calculate resolution ratio crop parameters
      const videoAspectRatio = videoWidth / videoHeight;
      const desiredAspectRatio = desiredWidth / desiredHeight;

      if (videoAspectRatio > desiredAspectRatio) {
        sHeight = videoHeight;
        sWidth = videoHeight * desiredAspectRatio;
        sx = (videoWidth - sWidth) / 2;
        sy = 0;
      } else {
        sWidth = videoWidth;
        sHeight = videoWidth / desiredAspectRatio;
        sx = 0;
        sy = (videoHeight - sHeight) / 2;
      }

      window.cropParams = { sx, sy, sWidth, sHeight };

      const scaleX = sWidth / desiredWidth;
      const scaleY = sHeight / desiredHeight;
      const scalingFactor = Math.min(scaleX, scaleY);
      window.globalScale = scalingFactor;

      if (scalingFactor < 1) {
        canvasElement.width = sWidth;
        canvasElement.height = sHeight;
      } else {
        canvasElement.width = desiredWidth;
        canvasElement.height = desiredHeight;
      }

      console.log(window.cropParams);

      detectFaces();
    };

  function detectFaces() {
    const canvasCtx = canvasElement.getContext('2d');
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
        if (selectedStyleName !== '' && selectedStyleData) {
          styleTransfer.getStylizedImage(canvasElement, landmarks_extracted, selectedStyleData);
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
  <video bind:this="{videoElement}" style="display: none;" aria-hidden="true"></video>
  <canvas bind:this="{canvasElement}"></canvas>
  <Styles onStyleSelected={handleStyleSelected} cropParams={window.cropParams}/>
</main>

<style>
  canvas {
    width: auto;
    height: 100%;
  }
</style>