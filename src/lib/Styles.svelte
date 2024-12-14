<script lang="ts">
  import { onMount } from 'svelte';
  import { MP2Dlib } from './MP2Dlib.ts';
  import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from '@mediapipe/tasks-vision';
  import { DB } from './Utils.ts';

  let stylesListFull = ['het', 'watercolorgirl', 'bronzestatue', 'charcoalman', 'expressive', 'frank', 'girl', 'illegalbeauty', 'ken', 'laurinbust', 'lincolnbust', 'malevich', 'oilman', 'oldman', 'prisma', 'stonebust', 'woodenmask'];
  let stylesList = ['het', 'watercolorgirl'];
  let stylesDataList: { [key: string]: any } = {};
  let selectedStyle = '';

  let isWebKit = false;

  const WIDTH = 480;
  const HEIGHT = 640;

  const db = new DB();
  const mp2dlib = new MP2Dlib();

  let faceLandmarker: FaceLandmarker | null = null;

  let loadingStates: { [key: string]: boolean } = {};

  export let stylesLoaded: (allFinished: boolean) => void = () => {};
  export let styleSelected: (styleName: string, styleData: any) => void = () => {};

  onMount(async () => {
    const ua = navigator.userAgent;
    isWebKit = /WebKit/.test(ua) && !/Edge/.test(ua) && !/Chrome/.test(ua);
    console.log('Detected WebKit:', isWebKit);
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: './assets/models/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: false,
      runningMode: 'IMAGE',
      numFaces: 1,
    });

    await Promise.all(stylesListFull.map(async (styleName) => {
      if (!isWebKit) {
        await loadSingleStyle(styleName);
      }
    }));
    stylesLoaded(await loadStyles());
  });

  async function loadStyles(): Promise<boolean> {
    let allFinished = true;
    await Promise.all(stylesList.map(async (styleName) => {
      if (stylesDataList[styleName]) {
        return;
      } else {
        console.log('Style:', styleName, 'is not cached, begin processing');
        if (!await addSingleStyle(styleName)) {
          allFinished = false;
        }
        if (!isWebKit) {
          await loadSingleStyle(styleName);
        }
      }
    }));
    return allFinished;
  }

  async function addSingleStyle(styleName: string): Promise<boolean> {
    try {
      console.log('Processing style:', styleName);

      const image = new Image();
      image.src = `./assets/styles/style_${styleName}_480x640.png`;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      ctx!.drawImage(image, 0, 0);
      const ImageData = ctx!.getImageData(0, 0, image.width, image.height);

      const lutArray = await loadLookUpCube(styleName);

      const results = faceLandmarker!.detect(canvas);
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = mp2dlib.transformLandmarks(results.faceLandmarks);
        const landmarksArray = transformLandmarksToInt32Array(landmarks[0]);
        // console.log('landmarksArray:', landmarksArray);

        const resizedBlob = await ImageDataToBlob(ImageData, 'image/png');

         if (isWebKit) {
          // Directly store in memory since IndexedDB is not available in WebKit
          const data = {
            resizedImage: ImageData,
            lookUpCube: lutArray,
            landmarksArray: landmarksArray
          };
          console.log('Data loaded direct to memory', styleName, data);
          stylesDataList[styleName] = data;
        } else {
          await db.saveStyleData(styleName, {
            resizedImage: resizedBlob,
            lookUpCube: lutArray.buffer,
            landmarksArray: landmarksArray.buffer,
          });
       }
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
    if (isWebKit) {
      return;
    }
    try {
      const data = await db.loadStyleData(styleName);
      if (data) {
        data.resizedImage = await blobToImageData(data.resizedImage);
        data.landmarksArray = new Int32Array(data.landmarksArray);
        data.lookUpCube = new Uint16Array(data.lookUpCube);
        stylesDataList[styleName] = data;
        console.log('Style:', styleName, 'is loaded from indexedDB');
      }
    } catch (error) {
      console.error('Error loading style:', styleName, error);
    }
  }

  async function loadLookUpCube(styleName: string): Promise<Uint16Array> {
    const fileName = `./assets/styles/lut_${styleName}_480x640.bytes`;
    try {
      const response = await fetch(fileName);
      if (!response.ok) {
        throw new Error(`Failed to load lookup cube from ${fileName}: ${response.status} ${response.statusText}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      const data = new Uint16Array(arrayBuffer);
      // console.log('Data for', styleName, data);
      return data;
    } catch (error) {
      console.error(`Error loading lookup cube: ${error.message}`);
      throw error;
    }
  }

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
    if (stylesDataList[styleName]) {
      selectedStyle = styleName;
      styleSelected(styleName, stylesDataList[styleName]);
    } else {
      console.warn('Style not loaded yet:', styleName);
    }
  }

  function transformLandmarksToInt32Array(landmarks: NormalizedLandmark[]): Int32Array {
    const landmarksArray = new Int32Array(landmarks.length * 2);
    for (let i = 0; i < landmarks.length; i++) {
      landmarksArray[i * 2] = Math.round(landmarks[i].x * WIDTH);
      landmarksArray[i * 2 + 1] = Math.round(landmarks[i].y * HEIGHT);
    }
    return landmarksArray;
  }

  async function handleStyleDownload(styleName: string) {
    if (!stylesDataList[styleName]) {
      loadingStates[styleName] = true;
      const success = await addSingleStyle(styleName);
      if (success && !isWebKit) {
        await loadSingleStyle(styleName);
      }
      loadingStates[styleName] = false;
    }
  }
</script>

<style>
  .grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: center;
  }
  .style-item {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    width: 150px;
    border: 2px solid transparent;
    padding: 8px;
    border-radius: 8px;
    transition: outline 0.3s;
  }
  .style-item img {
    width: 120px;
    object-fit: cover;
    border-radius: 4px;
  }
  .style-item:hover {
    outline: 2px solid rgb(63, 154, 211);
    outline-offset: 2px;
  }

  .style-item:hover * {
    outline: none;
  }
  .selected.style-item {
    outline: 2px solid rgb(63, 154, 211); 
    outline-offset: 2px;
  }

  .download-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.4);
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 24px;
    border-radius: 4px;
    cursor: pointer;
  }

  /* Spinner styles */
  .spinner {
    width: 32px;
    height: 32px;
    border: 4px solid #ffffff;
    border-top: 4px solid transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {transform: rotate(0deg);}
    to {transform: rotate(360deg);}
  }
</style>

<div class="grid">
  {#each stylesListFull as styleName}
    <div
      class="style-item {selectedStyle === styleName ? 'selected' : ''}"
    >
      <button
        type="button"
        on:click={() => {
          if (stylesDataList[styleName]) {
            selectStyle(styleName);
          } else {
            console.warn('Style data not available:', styleName);
          }
        }}
        aria-label={`Select style ${styleName}`}
      >
        <img src={`./assets/styles/style_${styleName}_480x640.png`} alt={styleName} />
      </button>

      <span>{styleName}</span>

      <!-- If style data not loaded yet, show either download icon or spinner if loading -->
      {#if !stylesDataList[styleName]}
        {#if loadingStates[styleName]}
          <div class="download-overlay">
            <div class="spinner"></div>
          </div>
        {:else}
          <button
            class="download-overlay"
            type="button"
            on:click={(e) => { e.stopPropagation(); handleStyleDownload(styleName); }}
            aria-label={`Download style ${styleName}`}
          >
            ⬇
          </button>
        {/if}
      {/if}
    </div>
  {/each}
</div>
