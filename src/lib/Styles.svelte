<script lang="ts">
  import { onMount } from 'svelte';
  import { StyleTransfer } from './StyleTransfer.ts';
  import { MP2Dlib } from './MP2Dlib.ts';
  import { FaceLandmarker, FilesetResolver,  DrawingUtils } from '@mediapipe/tasks-vision';
  import { DB } from './Utils.ts';
  let stylesList = ['het', 'watercolorgirl'];
  let stylesDataList = {};
  let selectedStyle = '';

  const styleTransfer = new StyleTransfer();
  const db = new DB();
  const mp2dlib = new MP2Dlib();

  let faceLandmarker: FaceLandmarker | null = null;
  let canvasElement: HTMLCanvasElement | null = null;

  export let onStyleSelected: (styleName: string, styleData: any) => void = () => {};
  export let cropParams: { sx: number; sy: number; sWidth: number; sHeight: number } = { sx: 0, sy: 0, sWidth: 0, sHeight: 0 };

  $: if (cropParams.sx !== 0 || cropParams.sy !== 0 || cropParams.sWidth !== 0 || cropParams.sHeight !== 0) {
    if (cropParams.sWidth > 768) {
      cropParams.sWidth = 768;
      cropParams.sHeight = 1024;
    }
    if (!loadStyles(cropParams)) {
      console.log('Error adding styles');
    }
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
      runningMode: 'IMAGE',
      numFaces: 1,
    });
  });

  function loadStyles(params: { sx: number; sy: number; sWidth: number; sHeight: number }): boolean {
    let allFinished = true;
    stylesList.forEach(async (styleName) => {
      await loadSingleStyle(styleName);
      // if is undefined, then add the style
      if (!stylesDataList[styleName]) {
        console.log('Style:', styleName, 'is not cached, begin processing');
        if (!addSingleStyle(styleName, params)) {
          allFinished = false;
        }
        await loadSingleStyle(styleName);
      }
    });
    return allFinished;
  }


  async function addSingleStyle(styleName: string, params: { sx: number; sy: number; sWidth: number; sHeight: number }): Promise<boolean> {
    try {
      console.log('Processing style:', styleName);

      // Load the style image
      const image = new Image();
      image.src = `/assets/styles/style_${styleName}.png`;
      await image.decode();

      // Resize the style image
      const resizedCanvas = styleTransfer.resizeImage(image, params.sWidth, params.sHeight);
      const resizedImageData = resizedCanvas.getContext('2d')!.getImageData(0, 0, params.sWidth, params.sHeight);

      // Compute the lookup cube
      const stylePosGuide = styleTransfer.getGradient(params.sWidth, params.sHeight, false);
      const styleAppGuide = styleTransfer.getAppGuide(resizedImageData, true);
      // const lutBuffer = await styleTransfer.getLookUpCube(stylePosGuide, styleAppGuide);
      const lutArray = await styleTransfer.getLookUpCubeCPUParallel(stylePosGuide, styleAppGuide);

      // // Perform face landmark detection
      const results = faceLandmarker!.detect(resizedCanvas);
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = mp2dlib.transformLandmarks(results.faceLandmarks);
        console.log('Landmarks:', landmarks);
        // Convert canvases to Base64 strings for storage
        const resizedBlob = await ImageDataToBlob(resizedImageData, 'image/png');
        const gradientBlob = await ImageDataToBlob(stylePosGuide, 'image/png');
        const appGuideBlob = await ImageDataToBlob(styleAppGuide, 'image/png');
      //   // Convert LUT buffer to Base64 string
      //   await lutBuffer.mapAsync(GPUMapMode.READ);
      //   const lutArray = new Uint8Array(lutBuffer.getMappedRange());
      //   const lutBase64 = btoa(String.fromCharCode(...lutArray));
      //   lutBuffer.unmap();

        // Save all data to IndexedDB
        await db.saveStyleData(styleName, {
          resizedImage: resizedBlob,
          gradient: gradientBlob,
          appGuide: appGuideBlob,
          lookupCube: lutArray.buffer,
          landmarks: landmarks,
        });
      } else {
        console.error('No landmarks detected.');
        return false;
      }

      console.log('Style processing complete:', styleName);
      return true;
    } catch (error) {
      console.error('Error processing style:', error);
      return false;
    }
  }

  async function loadSingleStyle(styleName: string) {
    try {
      const data = await db.loadStyleData(styleName);
      if (data) {
        // Convert Base64 strings back to canvases
        data.resizedImage = await blobToImageData(data.resizedImage);
        data.gradient = await blobToImageData(data.gradient);
        data.appGuide = await blobToImageData(data.appGuide);

        // Convert lookupCube ArrayBuffer back to Uint16Array
        data.lookupCube = new Uint16Array(data.lookupCube);
        stylesDataList[styleName] = data;
        console.log('Style:', styleName, 'is loaded');
      }
    } catch (error) {
      console.error('Error loading style:', error);
    }
  }

  // function canvasToBlob(canvas: HTMLCanvasElement, type: string): Promise<Blob> {
  //   return new Promise((resolve, reject) => {
  //     canvas.toBlob((blob) => {
  //       if (blob) {
  //         resolve(blob);
  //       } else {
  //         reject(new Error('Failed to convert canvas to Blob.'));
  //       }
  //     }, type);
  //   });
  // }

  function ImageDataToBlob(imageData: ImageData, type: string): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d');
      ctx!.putImageData(imageData, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert ImageData to Blob.'));
        }
      }, type);
    });
  }

  // function blobToCanvas(blob: Blob): Promise<HTMLCanvasElement> {
  //   return new Promise((resolve, reject) => {
  //     const canvas = document.createElement('canvas');
  //     const ctx = canvas.getContext('2d');
  //     const img = new Image();
  //     img.onload = () => {
  //       canvas.width = img.width;
  //       canvas.height = img.height;
  //       ctx!.drawImage(img, 0, 0);
  //       resolve(canvas);
  //     };
  //     img.onerror = (error) => {
  //       reject(error);
  //     };
  //     img.src = URL.createObjectURL(blob);
  //   });
  // }

  function blobToImageData(blob: Blob): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx!.drawImage(img, 0, 0);
        resolve(ctx!.getImageData(0, 0, img.width, img.height));
      };
      img.onerror = (error) => {
        reject(error);
      };
      img.src = URL.createObjectURL(blob);
    });
  }

  function selectStyle(styleName: string) {
    selectedStyle = styleName;
    onStyleSelected(styleName, stylesDataList[styleName]);
  }
</script>

<style>
  .grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }
  .style-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    width: 100px;
    border: 2px solid transparent;
    padding: 8px;
    border-radius: 8px;
    transition: border 0.3s;
  }
  .style-item:hover {
    border: 2px solid #ccc;
  }
  .style-item img {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 4px;
  }
  .selected.style-item {
    border: 2px solid white;
  }
</style>

<div class="grid">
  {#each stylesList as styleName}
    <button
      type="button"
      class="style-item {selectedStyle === styleName ? 'selected' : ''}"
      on:click={() => selectStyle(styleName)}
    >
      <img src={`/assets/styles/style_${styleName}.png`} alt={styleName} />
      <span>{styleName}</span>
    </button>
  {/each}
</div>

