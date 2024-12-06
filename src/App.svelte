<script>
  import { onMount } from 'svelte';
  import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

  let videoElement;
  let canvasElement;
  let faceLandmarker;

  onMount(async () => {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: '/assets/models/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: true,
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
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      detectFaces();
    };
  }

  function detectFaces() {
    const canvasCtx = canvasElement.getContext('2d');
    const drawingUtils = new DrawingUtils(canvasCtx);

    async function render() {
      const results = faceLandmarker.detectForVideo(videoElement, Date.now());

      canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

      if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: '#C0C0C070', lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: '#30FF30' }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: '#FF3030' }
          );
          // 添加更多绘制逻辑根据需要
        }
      }

      requestAnimationFrame(render);
    }

    render();
  }
</script>

<main>
  <video bind:this="{videoElement}" style="display: none;" aria-hidden="true"></video>
  <canvas bind:this="{canvasElement}"></canvas>
</main>

<style>
  canvas {
    width: 100%;
    height: auto;
  }
</style>