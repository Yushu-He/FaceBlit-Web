import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { ImgWarp_MLS_Similarity } from './ImgWarp_MLS_Similarity';
import type { Point } from './ImgWarp_MLS_Similarity';
import { CartesianCoordinateSystem } from './CartesianCoordinateSystem';

const GRID_SIZE = 10;
const MEAN = 128;
const RESIZE_RATIO_LM = 2;
const RESIZE_RATIO_MLS = 1;

// Parameters for the lookup cube
const lambdaPos = 1;
const lambdaApp = 1;
const radius = 30;
const cubeSize = 256;

const KERNEL = [1, 4, 6, 4, 1]; // Gaussian kernel for pyrDown
const KERNEL_SUM = 256; // (1+4+6+4+1)*(1+4+6+4+1) = 256

function clamp(value: number, minVal: number, maxVal: number): number {
    return Math.max(minVal, Math.min(value, maxVal));
}

function pyrDownOnce(data: Uint8ClampedArray, width: number, height: number): {data: Uint8ClampedArray, width: number, height: number} {
    const newWidth = Math.floor(width / 2);
    const newHeight = Math.floor(height / 2);
    const newData = new Uint8ClampedArray(newWidth * newHeight);

    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const centerX = x * 2;
            const centerY = y * 2;
            let sum = 0;

            // 5x5 neighborhood
            for (let i = -2; i <= 2; i++) {
                for (let j = -2; j <= 2; j++) {
                    const srcX = clamp(centerX + j, 0, width - 1);
                    const srcY = clamp(centerY + i, 0, height - 1);

                    const val = data[srcY * width + srcX];
                    const w = KERNEL[i + 2] * KERNEL[j + 2];
                    sum += val * w;
                }
            }

            const outVal = Math.round(sum / KERNEL_SUM);
            newData[y * newWidth + x] = outVal;
        }
    }

    return { data: newData, width: newWidth, height: newHeight };
}

function pyrDown(data: Uint8ClampedArray, width: number, height: number, levels: number): Uint8ClampedArray {
    let currentData = data;
    let currentWidth = width;
    let currentHeight = height;

    for (let level = 0; level < levels; level++) {
        const result = pyrDownOnce(currentData, currentWidth, currentHeight);
        currentData = result.data;
        currentWidth = result.width;
        currentHeight = result.height;
    }

    return currentData;
}

function landmarksToPoints(landmarks: NormalizedLandmark[], width: number, height: number): Point[] {
    return landmarks.map((lm) => {
        return {
            x: Math.round(lm.x * width),
            y: Math.round(lm.y * height),
        };
    });
}

// Histogram matching for grayscale images in TypeScript

function grayHistMatching(I: ImageData, R: ImageData): ImageData | null {
  // I and R are ImageData objects representing grayscale images

  const L = 256; // Number of intensity levels

  if (I.data.length !== R.data.length) {
    console.error("Input images must be of the same size");
    return null;
  }

  // Extract grayscale data from I and R
  const I_gray = extractGrayscale(I);
  const R_gray = extractGrayscale(R);

  // Compute histograms
  const I_hist = computeHistogram(I_gray, L);
  const R_hist = computeHistogram(R_gray, L);

  // Compute normalized histograms (PDF)
  const I_pdf = I_hist.map(count => count / I_gray.length);
  const R_pdf = R_hist.map(count => count / R_gray.length);

  // Compute cumulative distribution functions (CDF)
  const I_cdf = cumSum(I_pdf);
  const R_cdf = cumSum(R_pdf);

  // Scale CDFs to [0, L-1] and round
  const I_cdf_scaled = I_cdf.map(value => Math.round(value * (L - 1)));
  const R_cdf_scaled = R_cdf.map(value => Math.round(value * (L - 1)));

  // Create mapping from I to R
  const mapping = new Uint8Array(L);
  for (let i = 0; i < L; i++) {
    let minDiff = Infinity;
    let index = 0;
    for (let j = 0; j < L; j++) {
      const diff = Math.abs(I_cdf_scaled[i] - R_cdf_scaled[j]);
      if (diff < minDiff) {
        minDiff = diff;
        index = j;
      }
    }
    mapping[i] = index;
  }

  // Apply mapping to I
  const matchedGray = applyMapping(I_gray, mapping);

  // Reconstruct ImageData
  const resultImageData = new ImageData(I.width, I.height);
  for (let i = 0; i < matchedGray.length; i++) {
    const idx = i * 4;
    resultImageData.data[idx] = matchedGray[i];
    resultImageData.data[idx + 1] = matchedGray[i];
    resultImageData.data[idx + 2] = matchedGray[i];
    resultImageData.data[idx + 3] = 255; // Set alpha to opaque
  }

  return resultImageData;
}

function extractGrayscale(imageData: ImageData): Uint8ClampedArray {
  // Extract grayscale values from ImageData
  const gray = new Uint8ClampedArray(imageData.width * imageData.height);
  for (let i = 0; i < gray.length; i++) {
    const idx = i * 4;
    // Assuming grayscale image (R = G = B)
    gray[i] = imageData.data[idx];
  }
  return gray;
}

function computeHistogram(data: Uint8ClampedArray, L: number): number[] {
  const hist = new Array(L).fill(0);
  for (let i = 0; i < data.length; i++) {
    hist[data[i]]++;
  }
  return hist;
}

function cumSum(arr: number[]): number[] {
  const result = new Array(arr.length).fill(0);
  result[0] = arr[0];
  for (let i = 1; i < arr.length; i++) {
    result[i] = result[i - 1] + arr[i];
  }
  return result;
}

function applyMapping(data: Uint8ClampedArray, mapping: Uint8Array): Uint8ClampedArray {
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = mapping[data[i]];
  }
  return result;
}

// help function

function addPoints(p1: Point, p2: Point): Point {
    return { x: p1.x + p2.x, y: p1.y + p2.y };
}

function getPixelValue(imageData: ImageData, point: Point): [number, number, number, number] {
    const { x, y } = point;
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw new Error('Point is out of bounds');
    }
    const index = (y * width + x) * 4;
    return [
        data[index],       // R
        data[index + 1],   // G
        data[index + 2],   // B
        data[index + 3],   // A
    ];
}

function setPixelValue(imageData: ImageData, point: Point, pixel: [number, number, number, number]): void {
    const { x, y } = point;
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw new Error('Point is out of bounds');
    }
    const index = (y * width + x) * 4;
    data[index] = pixel[0];       // R
    data[index + 1] = pixel[1];   // G
    data[index + 2] = pixel[2];   // B
    data[index + 3] = pixel[3];   // A
}

function pixelOutOfImageRange(
    stylePixel: Point,
    targetPixel: Point,
    styleImage: ImageData,
    targetImage: ImageData
    ): boolean {
    return (
        stylePixel.x < 0 ||
        stylePixel.x >= styleImage.width ||
        stylePixel.y < 0 ||
        stylePixel.y >= styleImage.height ||
        targetPixel.x < 0 ||
        targetPixel.x >= targetImage.width ||
        targetPixel.y < 0 ||
        targetPixel.y >= targetImage.height
);
}

function pixelIsNotCovered(pixel: Point, coveredPixels: number[][]): boolean {
    return coveredPixels[pixel.y][pixel.x] === 0;
}

function getError(
    stylePosPixelG: number,
    stylePosPixelR: number,
    targetPosPixelG: number,
    targetPosPixelR: number,
    styleAppPixel: number,
    targetAppPixel: number,
    lambdaPos: number,
    lambdaApp: number
    ): number {
    const posErr = Math.abs(stylePosPixelG - targetPosPixelG) + Math.abs(stylePosPixelR - targetPosPixelR);
    const appErr = Math.abs(styleAppPixel - targetAppPixel);
    const totalError = posErr * lambdaPos + appErr * lambdaApp;
    return totalError;
}

// ----------------------------Skin Mask--------------------------------

function skinError(
    samplePixelCol: [number, number, number],
    currPixelCol: [number, number, number],
    use_YUV: boolean
  ): number {
    const colDiff = [
      samplePixelCol[0] - currPixelCol[0],
      samplePixelCol[1] - currPixelCol[1],
      samplePixelCol[2] - currPixelCol[2],
    ];
    let colDiffSumSquared = Math.pow(colDiff[1], 2) + Math.pow(colDiff[2], 2);
    if (!use_YUV) {
      colDiffSumSquared += Math.pow(colDiff[0], 2);
    }
    return colDiffSumSquared;
}

function sampleColors(imageData: ImageData, samplePoint: Point): [number, number, number] {
    const pointsToSample: Point[] = [
      { x: samplePoint.x - 5, y: samplePoint.y - 5 },
      { x: samplePoint.x - 5, y: samplePoint.y },
      { x: samplePoint.x - 5, y: samplePoint.y + 5 },
      { x: samplePoint.x, y: samplePoint.y - 5 },
      { x: samplePoint.x, y: samplePoint.y },
      { x: samplePoint.x, y: samplePoint.y + 5 },
      { x: samplePoint.x + 5, y: samplePoint.y - 5 },
      { x: samplePoint.x + 5, y: samplePoint.y },
      { x: samplePoint.x + 5, y: samplePoint.y + 5 },
    ];
  
    let accR = 0;
    let accG = 0;
    let accB = 0;
    let sampledColorsCount = 0;
  
    for (const pt of pointsToSample) {
      if (!CartesianCoordinateSystem.isPointInsideImage(pt, imageData)) {
        continue;
      }
      const sampledColor = getPixel(imageData, pt.x, pt.y);
      accR += sampledColor[0];
      accG += sampledColor[1];
      accB += sampledColor[2];
      sampledColorsCount++;
    }
  
    return [
      Math.round(accR / sampledColorsCount),
      Math.round(accG / sampledColorsCount),
      Math.round(accB / sampledColorsCount),
    ];
}

function getPixel(imageData: ImageData, x: number, y: number): [number, number, number] {
    const index = (y * imageData.width + x) * 4;
    return [
      imageData.data[index],     // R
      imageData.data[index + 1], // G
      imageData.data[index + 2], // B
    ];
}

function extractROI(
    imageData: ImageData,
    roi: { x: number; y: number; width: number; height: number }
  ): { data: Uint8ClampedArray; width: number; height: number } {
    const { x, y, width, height } = roi;
    const roiData = new Uint8ClampedArray(width * height * 4);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const srcIndex = ((row + y) * imageData.width + (col + x)) * 4;
        const dstIndex = (row * width + col) * 4;
        for (let i = 0; i < 4; i++) {
          roiData[dstIndex + i] = imageData.data[srcIndex + i];
        }
      }
    }
    return { data: roiData, width, height };
  }
  
  // Paste ROI data back into the result image data
  function pasteROI(
    resultData: Float32Array,
    resultWidth: number,
    roi: { x: number; y: number; width: number; height: number },
    roiData: Float32Array,
    roiWidth: number
  ): void {
    for (let row = 0; row < roi.height; row++) {
      const destRow = row + roi.y;
      for (let col = 0; col < roi.width; col++) {
        const destCol = col + roi.x;
        resultData[destRow * resultWidth + destCol] = roiData[row * roiWidth + col];
      }
    }
  }

// Convert RGB data to YUV
function RGBtoYUV(data: Uint8ClampedArray): Uint8ClampedArray {
    const yuvData = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

    const y = 16 + (0.257 * r + 0.504 * g + 0.098 * b);
    const u = 128 + (-0.148 * r - 0.291 * g + 0.439 * b);
    const v = 128 + (0.439 * r - 0.368 * g - 0.071 * b);

      yuvData[i] = Math.round(y);
      yuvData[i + 1] = Math.round(u);
      yuvData[i + 2] = Math.round(v);
      yuvData[i + 3] = data[i + 3];
    }
    return yuvData;
}

  
  // Fill a convex polygon in the image mask
  function fillConvexPoly(
    maskData: Float32Array,
    width: number,
    height: number,
    points: Point[]
  ): void {
    // Simple scan-line fill algorithm
    const polygon = points.map((pt) => ({ x: Math.round(pt.x), y: Math.round(pt.y) }));
    const minY = Math.max(
      Math.min(...polygon.map((pt) => pt.y)),
      0
    );
    const maxY = Math.min(
      Math.max(...polygon.map((pt) => pt.y)),
      height - 1
    );
  
    for (let y = minY; y <= maxY; y++) {
      const nodes: number[] = [];
      let j = polygon.length - 1;
      for (let i = 0; i < polygon.length; i++) {
        if (polygon[i].y < y && polygon[j].y >= y || polygon[j].y < y && polygon[i].y >= y) {
          const x =
            polygon[i].x +
            ((y - polygon[i].y) * (polygon[j].x - polygon[i].x)) /
              (polygon[j].y - polygon[i].y);
          nodes.push(x);
        }
        j = i;
      }
      nodes.sort((a, b) => a - b);
      for (let k = 0; k < nodes.length; k += 2) {
        if (nodes[k] >= width) break;
        if (nodes[k + 1] > 0) {
          const startX = Math.max(Math.round(nodes[k]), 0);
          const endX = Math.min(Math.round(nodes[k + 1]), width - 1);
          for (let x = startX; x <= endX; x++) {
            maskData[y * width + x] = 1;
          }
        }
      }
    }
  }

  function fillEllipse(
    maskData: Float32Array,
    width: number,
    height: number,
    center: Point,
    axes: { x: number; y: number }
  ): void {
    const minX = Math.max(0, Math.floor(center.x - axes.x));
    const maxX = Math.min(width - 1, Math.ceil(center.x + axes.x));
    const minY = Math.max(0, Math.floor(center.y - axes.y));
    const maxY = Math.min(height - 1, Math.ceil(center.y + axes.y));
  
    const ax2 = axes.x * axes.x;
    const ay2 = axes.y * axes.y;
  
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - center.x;
        const dy = y - center.y;
        if ((dx * dx) / ax2 + (dy * dy) / ay2 <= 1) {
          maskData[y * width + x] = 1;
        }
      }
    }
  }

  function boxBlur(
    data: Float32Array,
    width: number,
    height: number,
    kernelSize: number
  ): Float32Array {
    const halfKernel = Math.floor(kernelSize / 2);
    const result = new Float32Array(data.length);
  
    // Assume data has 4 channels
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const sum = [0, 0, 0, 0];
        let count = 0;
  
        for (let ky = -halfKernel; ky <= halfKernel; ky++) {
          const ny = y + ky;
          if (ny >= 0 && ny < height) {
            for (let kx = -halfKernel; kx <= halfKernel; kx++) {
              const nx = x + kx;
              if (nx >= 0 && nx < width) {
                const idx = (ny * width + nx) * 4;
                sum[0] += data[idx];
                sum[1] += data[idx + 1];
                sum[2] += data[idx + 2];
                sum[3] += data[idx + 3];
                count++;
              }
            }
          }
        }
  
        const idx = (y * width + x) * 4;
        result[idx] = sum[0] / count;
        result[idx + 1] = sum[1] / count;
        result[idx + 2] = sum[2] / count;
        result[idx + 3] = sum[3] / count;
      }
    }
  
    return result;
  }

// let device: GPUDevice | null = null;

export class StyleTransfer {

  constructor() {
    // if (device === null) {
    //   this.initWebGPU();
    // }
  }

  resizeImage(image: HTMLImageElement, width: number, height: number): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Failed to get canvas context');
        return canvas;
    }

    ctx.drawImage(image, 0, 0, width, height);
    return canvas;
  }
  
  getGradient(width: number, height: number, drawGrid: boolean): ImageData {
    const widthNorm = 256 / width;
    const heightNorm = 256 / height;

    const imageData = new ImageData(width, height);
    const data = imageData.data;

    for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
            const index = (row * width + col) * 4; // RGBA format

            if (drawGrid && ((row > 1 && row % GRID_SIZE === 0) || (col > 1 && col % GRID_SIZE === 0))) {
                // White grid
                data[index] = 255; // Red
                data[index + 1] = 255; // Green
                data[index + 2] = 255; // Blue
                data[index + 3] = 255; // Alpha
            } else {
                // Gradient
                data[index] = col * widthNorm; // Red channel
                data[index + 1] = row * heightNorm; // Green channel
                data[index + 2] = 0; // Blue channel
                data[index + 3] = 255; // Alpha channel
            }
        }
    }

    return imageData;
  }


  getAppGuide(imageData: ImageData, stretchHist: boolean): ImageData {
    const width = imageData.width;
    const height = imageData.height;
    const grayData = new Uint8ClampedArray(width * height);

    // Convert to grayscale
    for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        grayData[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // Downscale using pyrDown
    const downscaleFactor = Math.floor(width / 256);
    const blurData = pyrDown(grayData, width, height, downscaleFactor);

    // Upscale back to original size using bilinear interpolation
    const resizedBlurData = this.upscaleBilinear(blurData, width, height, downscaleFactor);

    let minVal = 255;
    let maxVal = 0;
    const resultData = new Uint8ClampedArray(grayData.length);

    // Compute G_app
    for (let i = 0; i < grayData.length; i++) {
        const diff = grayData[i] - resizedBlurData[i];
        const val = Math.trunc(diff / 2) + MEAN;
        const clamped = Math.max(0, Math.min(255, val));
        resultData[i] = clamped;

        if (clamped < minVal) minVal = clamped;
        if (clamped > maxVal) maxVal = clamped;
    }

    if (stretchHist) {
        // Symmetrical histogram stretch
        const x = Math.min(minVal, 255 - maxVal);
        const newMin = x;
        const newMax = 255 - x;
        const range = newMax - newMin;

        for (let i = 0; i < resultData.length; i++) {
            const stretched = ((resultData[i] - newMin) / range) * 255;
            resultData[i] = Math.round(Math.max(0, Math.min(255, stretched)));
        }
    }

    const outputImageData = new ImageData(width, height);
    for (let i = 0; i < resultData.length; i++) {
        const v = resultData[i];
        outputImageData.data[i * 4] = v; // Red
        outputImageData.data[i * 4 + 1] = v; // Green
        outputImageData.data[i * 4 + 2] = v; // Blue
        outputImageData.data[i * 4 + 3] = 255; // Alpha
    }

    return outputImageData;
  }


  downscale(data: Uint8ClampedArray, width: number, height: number, levels: number): Uint8ClampedArray {
    let currentData = data;
    let currentWidth = width;
    let currentHeight = height;

    for (let level = 0; level < levels; level++) {
        const newWidth = Math.floor(currentWidth / 2);
        const newHeight = Math.floor(currentHeight / 2);
        const newData = new Uint8ClampedArray(newWidth * newHeight);

        for (let y = 0; y < newHeight; y++) {
            for (let x = 0; x < newWidth; x++) {
                const topLeft = currentData[(y * 2) * currentWidth + (x * 2)];
                const topRight = (x * 2 + 1 < currentWidth) 
                    ? currentData[(y * 2) * currentWidth + (x * 2 + 1)] 
                    : topLeft;
                const bottomLeft = (y * 2 + 1 < currentHeight) 
                    ? currentData[(y * 2 + 1) * currentWidth + (x * 2)] 
                    : topLeft;
                const bottomRight = (y * 2 + 1 < currentHeight && x * 2 + 1 < currentWidth)
                    ? currentData[(y * 2 + 1) * currentWidth + (x * 2 + 1)]
                    : bottomLeft;

                const sum = (topLeft + topRight + bottomLeft + bottomRight) / 4;
                newData[y * newWidth + x] = Math.round(sum);
            }
        }

        currentData = newData;
        currentWidth = newWidth;
        currentHeight = newHeight;
    }

    return currentData;
  }

  // Bilinear upscale to match cv::INTER_LINEAR
  upscaleBilinear(data: Uint8ClampedArray, outWidth: number, outHeight: number, levels: number): Uint8ClampedArray {
    const scale = Math.pow(2, levels); // after downscale by pyrDown levels times, dimension = original/2^levels
    const inWidth = Math.floor(outWidth / scale);
    const inHeight = Math.floor(outHeight / scale);

    const output = new Uint8ClampedArray(outWidth * outHeight);

    const scaleX = (inWidth - 1) / (outWidth - 1);
    const scaleY = (inHeight - 1) / (outHeight - 1);

    for (let y = 0; y < outHeight; y++) {
        const srcY = y * scaleY;
        const y1 = Math.floor(srcY);
        const y2 = Math.min(y1 + 1, inHeight - 1);
        const fy = srcY - y1;

        for (let x = 0; x < outWidth; x++) {
            const srcX = x * scaleX;
            const x1 = Math.floor(srcX);
            const x2 = Math.min(x1 + 1, inWidth - 1);
            const fx = srcX - x1;

            const topLeft = data[y1 * inWidth + x1];
            const topRight = data[y1 * inWidth + x2];
            const bottomLeft = data[y2 * inWidth + x1];
            const bottomRight = data[y2 * inWidth + x2];

            const val = (topLeft * (1 - fx) * (1 - fy)) +
                        (topRight * fx * (1 - fy)) +
                        (bottomLeft * (1 - fx) * fy) +
                        (bottomRight * fx * fy);

            output[y * outWidth + x] = Math.round(val);
        }
    }

    return output;
  }

  async getLookUpCubeCPUParallel(
    stylePosGuide: ImageData,       // RGBA, R/G用于pos guide, B/A未用
    styleAppGuide: ImageData,       // RGBA, 假设R通道是app guide灰度(根据需要调整)
    numWorkers: number = 8,
    lambdaPos: number = 10,
    lambdaApp: number = 2
  ): Promise<Uint16Array> {
    console.log('Computing lookup cube using CPU parallel');
    const stylePosWidth = stylePosGuide.width;
    const stylePosHeight = stylePosGuide.height;
    const styleAppWidth = styleAppGuide.width;
    const styleAppHeight = styleAppGuide.height;

    // 从ImageData中分离出用于计算的数据
    // stylePosGuide: RGBA数组，每个像素4字节: [R,G,B,A]
    // 这里R = pos红通道, G = pos绿通道; 对应原始C++中的val[1], val[2]只是顺序可能不同，需要根据需要调整
    // 如原C++中使用 val[1], val[2] (G,R) 那么这里相当于 posG= data[index+1], posR= data[index]
    // 假设保持一致: posG对应G通道(data[index+1])，posR对应R通道(data[index])。
    // appGuide使用styleAppGuide R通道作为灰度值。
    const stylePosGuideData = new Uint8Array(stylePosGuide.data.buffer);
    const styleAppGuideData = new Uint8Array(styleAppGuide.data.buffer);

    // 创建SharedArrayBuffer并拷贝数据进去
    // posGuide是4通道，appGuide是4通道但我们只需其中一个通道，这里先全部复制，worker中再访问需要的通道
    const posSAB = new SharedArrayBuffer(stylePosGuideData.byteLength);
    const appSAB = new SharedArrayBuffer(styleAppGuideData.byteLength);

    const posShared = new Uint8Array(posSAB);
    posShared.set(stylePosGuideData);

    const appShared = new Uint8Array(appSAB);
    appShared.set(styleAppGuideData);

    // 输出结果 lookUpCube: 256x256x256，每个元素存2个uint16( col,row )
    // 大小 = 256*256*256*2
    const lookUpCube = new Uint16Array(cubeSize * cubeSize * cubeSize * 2);

    const xNorm = stylePosWidth / 256.0;
    const yNorm = stylePosHeight / 256.0;

    // 创建Worker脚本
    const workerScript = `
    onmessage = function(e) {
      const {
        startX, endX,
        posSAB, appSAB,
        stylePosWidth, stylePosHeight,
        styleAppWidth, styleAppHeight,
        lambdaPos, lambdaApp, xNorm, yNorm, cubeSize, radius
      } = e.data;
      
      // TypedArray视图在worker中创建
      const posData = new Uint8Array(posSAB);
      const appData = new Uint8Array(appSAB);

      function getError(stylePosPixelG, stylePosPixelR, targetPosPixelG, targetPosPixelR, styleAppPixel, targetAppPixel, LAMBDA_POS, LAMBDA_APP) {
        const posErr = Math.abs(stylePosPixelG - targetPosPixelG) + Math.abs(stylePosPixelR - targetPosPixelR);
        const appErr = Math.abs(styleAppPixel - targetAppPixel);
        const totalError = posErr * LAMBDA_POS + appErr * LAMBDA_APP;
        return totalError;
      }

      // 计算输出大小
      // 对于每个worker计算从startX到endX(不包含endX)范围内的cube
      // 范围长度: (endX - startX)
      // 每个x层有 cubeSize * cubeSize * 2 (uint16)
      const widthXRange = endX - startX;
      const result = new Uint16Array(widthXRange * cubeSize * cubeSize * 2);

      for (let localX = startX; localX < endX; localX++) {
        const x = localX;
        for (let y = 0; y < cubeSize; y++) {
          const seedRow = Math.floor(y * yNorm);
          const seedCol = Math.floor(x * xNorm);

          for (let z = 0; z < cubeSize; z++) {
            let minError = Infinity;
            let bestCol = seedCol;
            let bestRow = seedRow;

            // 在posData中:
            // 每个像素: RGBA => index = (row*stylePosWidth+col)*4
            // posR = data[index], posG = data[index+1]
            // 在appData中:
            // app通道假设使用R通道: appVal = appData[(row*styleAppWidth+col)*4]
            for (let row = Math.max(0, seedRow - radius); row < Math.min(stylePosHeight, seedRow + radius); row++) {
              for (let col = Math.max(0, seedCol - radius); col < Math.min(stylePosWidth, seedCol + radius); col++) {
                const posIndex = (row * stylePosWidth + col)*4;
                const posValR = posData[posIndex];     // R
                const posValG = posData[posIndex + 1]; // G
                const appVal = appData[(row * styleAppWidth + col)*4]; // R通道作为app

                const error = getError(
                  posValG, posValR,
                  y, x,
                  appVal, z,
                  lambdaPos, lambdaApp
                );
                if (error < minError) {
                  minError = error;
                  bestCol = col;
                  bestRow = row;
                }
              }
            }

            const offsetX = localX - startX;
            const idx = ((offsetX * cubeSize + y) * cubeSize + z)*2;
            result[idx] = bestCol;
            result[idx+1] = bestRow;
          }
        }
      }

      postMessage({startX, endX, result}, [result.buffer]);
    };
    `;

    const workerScriptUrl = URL.createObjectURL(new Blob([workerScript], { type: 'application/javascript' }));
    const workers: Worker[] = [];
    for (let i = 0; i < numWorkers; i++) {
      workers.push(new Worker(workerScriptUrl));
    }

    const blockSize = Math.ceil(cubeSize / numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < numWorkers; i++) {
      const startX = i * blockSize;
      const endX = Math.min((i+1)*blockSize, cubeSize);

      if (startX >= cubeSize) break;
      const promise = new Promise<void>((resolve) => {
        workers[i].onmessage = (e) => {
          const {startX, endX, result} = e.data;
          // 将结果拷贝回lookUpCube
          for (let localX = startX; localX < endX; localX++) {
            const offsetX = localX - startX;
            lookUpCube.set(result.subarray(offsetX * cubeSize * cubeSize * 2, (offsetX+1)*cubeSize*cubeSize*2), localX * cubeSize * cubeSize * 2);
          }
          resolve();
        };
      });

      workers[i].postMessage({
        startX,
        endX,
        posSAB,
        appSAB,
        stylePosWidth, stylePosHeight,
        styleAppWidth, styleAppHeight,
        lambdaPos,
        lambdaApp,
        xNorm,
        yNorm,
        cubeSize,
        radius
      });

      promises.push(promise);
    }

    await Promise.all(promises);
    workers.forEach(w => w.terminate());
    URL.revokeObjectURL(workerScriptUrl);

    console.log('Lookup cube computed');
    return lookUpCube;
  }

  MLSDeformation(
      gradientImgData: ImageData,
      stylelandmarks: Point[],
      targetlandmarks: Point[]
  ): ImageData {
    const imgTrans = new ImgWarp_MLS_Similarity();
    imgTrans.alpha = 1.0; // 0.1 - 3.0, default 1.0, step 0.1
    imgTrans.gridSize = GRID_SIZE; // 1 - 20, default 5

    return imgTrans.setAllAndGenerate(
        gradientImgData,
        stylelandmarks,
        targetlandmarks,
        gradientImgData.width,
        gradientImgData.height,
        RESIZE_RATIO_MLS
    );
  }


  DFSSeedGrow(
    targetSeedPoint: Point,
    styleSeedPoint: Point,
    stylePosGuide: ImageData,
    targetPosGuide: ImageData,
    styleAppGuide: ImageData | null,
    targetAppGuide: ImageData | null,
    resultImg: ImageData,
    styleImg: ImageData,
    coveredPixels: number[][],
    chunkNumber: number,
    threshold: number,
    lambdaPos: number,
    lambdaApp: number
  ): void {
    const offsetQueue: Point[] = [];
    offsetQueue.push({ x: 0, y: 0 });
  
    while (offsetQueue.length > 0) {
      const offset = offsetQueue.shift() as Point;
  
      const targetPoint = addPoints(targetSeedPoint, offset);
      const stylePoint = addPoints(styleSeedPoint, offset);
  
      if (
        pixelOutOfImageRange(
          stylePoint,
          targetPoint,
          stylePosGuide,
          targetPosGuide
        )
      ) {
        continue;
      }
  
      if (pixelIsNotCovered(targetPoint, coveredPixels)) {
        try {
          const [stylePosR, stylePosG] = getPixelValue(stylePosGuide, stylePoint);
          const [targetPosR, targetPosG] = getPixelValue(targetPosGuide, targetPoint);
  
          const styleAppPixel = styleAppGuide
            ? getPixelValue(styleAppGuide, stylePoint)[0]
            : 0;
          const targetAppPixel = targetAppGuide
            ? getPixelValue(targetAppGuide, targetPoint)[0]
            : 0;
  
          const error = getError(
            stylePosG,
            stylePosR,
            targetPosG,
            targetPosR,
            styleAppPixel,
            targetAppPixel,
            lambdaPos,
            lambdaApp
          );
  
          if (error < threshold || (offset.x === 0 && offset.y === 0)) {
            const stylePixel = getPixelValue(styleImg, stylePoint);
            setPixelValue(resultImg, targetPoint, stylePixel);
  
            coveredPixels[targetPoint.y][targetPoint.x] = chunkNumber;
  
            const left = { x: offset.x - 1, y: offset.y };
            const right = { x: offset.x + 1, y: offset.y };
            const up = { x: offset.x, y: offset.y - 1 };
            const down = { x: offset.x, y: offset.y + 1 };
  
            if (
              !pixelOutOfImageRange(
                addPoints(styleSeedPoint, left),
                addPoints(targetSeedPoint, left),
                stylePosGuide,
                targetPosGuide
              ) &&
              pixelIsNotCovered(addPoints(targetSeedPoint, left), coveredPixels)
            ) {
              offsetQueue.push(left);
            }
            if (
              !pixelOutOfImageRange(
                addPoints(styleSeedPoint, right),
                addPoints(targetSeedPoint, right),
                stylePosGuide,
                targetPosGuide
              ) &&
              pixelIsNotCovered(addPoints(targetSeedPoint, right), coveredPixels)
            ) {
              offsetQueue.push(right);
            }
            if (
              !pixelOutOfImageRange(
                addPoints(styleSeedPoint, up),
                addPoints(targetSeedPoint, up),
                stylePosGuide,
                targetPosGuide
              ) &&
              pixelIsNotCovered(addPoints(targetSeedPoint, up), coveredPixels)
            ) {
              offsetQueue.push(up);
            }
            if (
              !pixelOutOfImageRange(
                addPoints(styleSeedPoint, down),
                addPoints(targetSeedPoint, down),
                stylePosGuide,
                targetPosGuide
              ) &&
              pixelIsNotCovered(addPoints(targetSeedPoint, down), coveredPixels)
            ) {
              offsetQueue.push(down);
            }
          }
        } catch (error) {
          console.error('Error accessing pixel data:', error);
        }
      }
    }
  }

  styleBlit(
    stylePosGuide: ImageData,
    targetPosGuide: ImageData,
    styleAppGuide: ImageData | null,
    targetAppGuide: ImageData | null,
    lookUpCube: Uint16Array, // Assuming lookUpCube is a flat Uint16Array
    styleImg: ImageData,
    stylizationRangeRect: { x: number; y: number; width: number; height: number },
    threshold: number,
    lambdaPos: number,
    lambdaApp: number
  ): ImageData {
    const boxSize = 25;
    const width = styleImg.width;
    const height = styleImg.height;
    const resultImg = new ImageData(width, height);
    const coveredPixels: number[][] = Array.from({ length: height }, () => Array(width).fill(0)); // Helper to record already stylized pixels
    const xNorm = width / 256.0;
    const yNorm = height / 256.0;
    let chunkNumber = 1; // For visualization of chunks

  
    // Iterate over the specified stylization range
    const startY = Math.floor(stylizationRangeRect.y);
    const endY = Math.floor(stylizationRangeRect.y + stylizationRangeRect.height);
    const startX = Math.floor(stylizationRangeRect.x);
    const endX = Math.floor(stylizationRangeRect.x + stylizationRangeRect.width);
  
    for (let rowT = startY; rowT < endY; rowT++) {
      for (let colT = startX; colT < endX; colT++) {
        if (coveredPixels[rowT][colT] === 0) {
          let styleSeedPoint: Point;
  
          if (lambdaApp === 0) {
            // Without appearance guide
            const targetPosPixel = getPixelValue(targetPosGuide, { x: colT, y: rowT });
            styleSeedPoint = {
              x: Math.floor(targetPosPixel[2] * xNorm), // R channel
              y: Math.floor(targetPosPixel[1] * yNorm), // G channel
            };
          } else {
            // Retrieve the style seed point from the lookup cube
            const targetPosPixel = getPixelValue(targetPosGuide, { x: colT, y: rowT });
            const appValue = targetAppGuide ? getPixelValue(targetAppGuide, { x: colT, y: rowT })[0] : 0;
  
            const posR = targetPosPixel[2] & 0xff;
            const posG = targetPosPixel[1] & 0xff;
            const app = appValue & 0xff;
  
            // Calculate cube index
            const cubeIndex = ((posR * 256 * 256) + (posG * 256) + app) * 2;
  
            const styleCol = lookUpCube[cubeIndex];
            const styleRow = lookUpCube[cubeIndex + 1];
            styleSeedPoint = { x: styleCol, y: styleRow };
          }
  
          // Perform DFS seed growing
          this.DFSSeedGrow(
            { x: colT, y: rowT },
            styleSeedPoint,
            stylePosGuide,
            targetPosGuide,
            styleAppGuide,
            targetAppGuide,
            resultImg,
            styleImg,
            coveredPixels,
            chunkNumber,
            threshold,
            lambdaPos,
            lambdaApp
          );
          chunkNumber++;
        }
      }
    }
  
    // The resultImg now contains the stylized image
    return resultImg;
  }

  getSkinMask(imageData: ImageData, landmarks: Point[]): ImageData {
    const faceContourPoints = landmarks.slice(0, 17);
    const faceWidth = faceContourPoints[16].x - faceContourPoints[0].x;
  
    const foreheadROI = {
      x: faceContourPoints[0].x,
      y: Math.max(faceContourPoints[0].y - Math.round(faceWidth * 0.75), 0),
      width: faceWidth,
      height: Math.min(Math.round(faceWidth * 0.75), faceContourPoints[0].y),
    };
  
    const foreheadRect = extractROI(imageData, foreheadROI);
    const samplePoint1 = {
      x: Math.round((faceWidth / 4) * 1),
      y: Math.max(foreheadRect.height - Math.round(faceWidth / 4), 0),
    };
    const samplePoint2 = {
      x: Math.round((faceWidth / 4) * 2),
      y: Math.max(foreheadRect.height - Math.round(faceWidth / 4), 0),
    };
    const samplePoint3 = {
      x: Math.round((faceWidth / 4) * 3),
      y: Math.max(foreheadRect.height - Math.round(faceWidth / 4), 0),
    };
  
    const USE_YUV = true;
    const SKIN_ERROR_THRESHOLD = 80;
  
    let foreheadData = foreheadRect.data;
    if (USE_YUV) {
      foreheadData = RGBtoYUV(foreheadData);
    }
  
    const sampleColor1 = sampleColors(
      new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
      samplePoint1
    );
    const sampleColor2 = sampleColors(
      new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
      samplePoint2
    );
    const sampleColor3 = sampleColors(
      new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
      samplePoint3
    );
  
    const resultForehead = new Float32Array(foreheadRect.width * foreheadRect.height);
  
    // Process sample point 1
    for (let row = 0; row < foreheadRect.height; row++) {
      for (let col = 0; col < Math.round((faceWidth / 4) * 2); col++) {
        const currPixel = getPixel(
          new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
          col,
          row
        );
        const error = skinError(sampleColor1, currPixel, USE_YUV);
        if (error < SKIN_ERROR_THRESHOLD) {
          resultForehead[row * foreheadRect.width + col] = 1;
        }
      }
    }
  
    // Process sample point 2
    for (let row = 0; row < foreheadRect.height; row++) {
      for (
        let col = Math.round(faceWidth / 4);
        col < Math.round((faceWidth / 4) * 3);
        col++
      ) {
        const currPixel = getPixel(
          new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
          col,
          row
        );
        const error = skinError(sampleColor2, currPixel, USE_YUV);
        if (error < SKIN_ERROR_THRESHOLD) {
          resultForehead[row * foreheadRect.width + col] = 1;
        }
      }
    }
  
    // Process sample point 3
    for (let row = 0; row < foreheadRect.height; row++) {
      for (let col = Math.round((faceWidth / 4) * 2); col < foreheadRect.width; col++) {
        const currPixel = getPixel(
          new ImageData(foreheadData, foreheadRect.width, foreheadRect.height),
          col,
          row
        );
        const error = skinError(sampleColor3, currPixel, USE_YUV);
        if (error < SKIN_ERROR_THRESHOLD) {
          resultForehead[row * foreheadRect.width + col] = 1;
        }
      }
    }
  
    // Create result image data
    const resultImgData = new Float32Array(imageData.width * imageData.height);
    pasteROI(resultImgData, imageData.width, foreheadROI, resultForehead, foreheadRect.width);
  
    // Fill face contour
    fillConvexPoly(resultImgData, imageData.width, imageData.height, faceContourPoints);
  
    const center: Point = {
        x: faceContourPoints[0].x + (faceContourPoints[16].x - faceContourPoints[0].x) / 2,
        y: faceContourPoints[0].y + (faceContourPoints[16].y - faceContourPoints[0].y) / 2,
      };
      const axes = {
        x: (faceContourPoints[16].x - faceContourPoints[0].x) / 2,
        y: (faceContourPoints[16].x - faceContourPoints[0].x) / 2.5,
      };
    fillEllipse(resultImgData, imageData.width, imageData.height, center, axes);

    return new ImageData(
        new Uint8ClampedArray(Array.from(resultImgData).flatMap(val => [
            val > 0 ? 255 : 0, // R
            val > 0 ? 255 : 0, // G
            val > 0 ? 255 : 0, // B
            255                // A
        ])),
        imageData.width,
        imageData.height
    );
  }

  alphaBlendFG_BG(
    foreground: ImageData,
    background: ImageData,
    alpha: ImageData,
    sigma: number
  ): ImageData {
    const width = foreground.width;
    const height = foreground.height;
    const size = width * height;
    const kernelSize = 20; // For box blur
  
    // Convert images to Float32Array and normalize to [0,1]
    const fgFloat = new Float32Array(foreground.data.length);
    const bgFloat = new Float32Array(background.data.length);
    for (let i = 0; i < foreground.data.length; i++) {
      fgFloat[i] = foreground.data[i] / 255;
      bgFloat[i] = background.data[i] / 255;
    }
  
    // Convert alpha to Float32Array and normalize
    let alphaFloat: Float32Array;
    if (alpha.data.length === width * height * 4) {
      // Alpha has 4 channels
      alphaFloat = new Float32Array(alpha.data.length);
      for (let i = 0; i < alpha.data.length; i++) {
        alphaFloat[i] = alpha.data[i] / 255;
      }
    } else if (alpha.data.length === width * height) {
      // Alpha is single-channel
      alphaFloat = new Float32Array(width * height * 4);
      for (let i = 0; i < size; i++) {
        const val = alpha.data[i] / 255;
        alphaFloat[i * 4] = val;
        alphaFloat[i * 4 + 1] = val;
        alphaFloat[i * 4 + 2] = val;
        alphaFloat[i * 4 + 3] = 1; // Set alpha channel to opaque
      }
    } else {
      throw new Error('Alpha mask has incompatible dimensions.');
    }
  
    // Apply blur to the alpha mask
    const blurredAlpha = boxBlur(alphaFloat, width, height, kernelSize);
  
    // Multiply alpha with foreground
    const fgAlpha = new Float32Array(fgFloat.length);
    for (let i = 0; i < fgFloat.length; i++) {
      fgAlpha[i] = fgFloat[i] * blurredAlpha[i];
    }
  
    // Multiply (1 - alpha) with background
    const bgAlpha = new Float32Array(bgFloat.length);
    for (let i = 0; i < bgFloat.length; i++) {
      bgAlpha[i] = bgFloat[i] * (1 - blurredAlpha[i]);
    }
  
    // Add the masked foreground and background
    const resultFloat = new Float32Array(fgAlpha.length);
    for (let i = 0; i < resultFloat.length; i++) {
      resultFloat[i] = fgAlpha[i] + bgAlpha[i];
    }
  
    // Convert the result back to Uint8ClampedArray
    const resultData = new Uint8ClampedArray(resultFloat.length);
    for (let i = 0; i < resultFloat.length; i++) {
      resultData[i] = Math.round(resultFloat[i] * 255);
    }
  
    return new ImageData(resultData, width, height);
  }

  async getStylizedImage(imageCanvas: HTMLCanvasElement, imageLandmarks: NormalizedLandmark[][], styleData: any){
    const targetImageData = imageCanvas.getContext('2d')!.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
    const targetlandmarksPoints = landmarksToPoints(imageLandmarks[0], targetImageData.width, targetImageData.height);
    const stylelandmarksPoints = landmarksToPoints(styleData.landmarks[0], targetImageData.width, targetImageData.height);
    const targetPosGuide = this.MLSDeformation(targetImageData, stylelandmarksPoints, targetlandmarksPoints);
    const targetAppGuideNoHist = this.getAppGuide(targetPosGuide, false);
    const targetAppGuide = grayHistMatching(targetAppGuideNoHist, styleData.appGuide);
    // -------------------------Syle Transfer-------------------------
    const cartesianCoordinateSystem = new CartesianCoordinateSystem();
    const headAreaRect = cartesianCoordinateSystem.getHeadAreaRect(targetlandmarksPoints, {width: targetImageData.width, height: targetImageData.height});
    const stylelizedImageData = this.styleBlit(styleData.gradient, targetPosGuide, styleData.appGuide, targetAppGuide, styleData.lookupCube, styleData.resizedImage, headAreaRect, 50, 10, 2);
    const targetFaceMask = this.getSkinMask(targetImageData, targetlandmarksPoints);

    // -------------------------Alpha Blending-------------------------
    // const alphaBlendImageData = this.alphaBlendFG_BG(stylelizedImageData, targetImageData, targetFaceMask, 25.0);

    const ctx = imageCanvas.getContext('2d');
    ctx!.putImageData(stylelizedImageData, 0, 0);
  }

//     // Initialize WebGPU
//   async initWebGPU() {
//     if (!navigator.gpu) {
//         console.error('WebGPU is not supported in this browser.');
//         return;
//     }

//     try {
//         const adapter = await navigator.gpu.requestAdapter();
//         if (!adapter) {
//         console.error('Failed to get GPU adapter.');
//         return;
//         }
//         device = await adapter.requestDevice();
//         console.log('WebGPU initialized.');
//     } catch (error) {
//         console.error('WebGPU initialization error:', error);
//     }
//   }

//   // Get lookup cube using WebGPU
//   async getLookUpCube(stylePosGuide: ImageData, styleAppGuide: ImageData): Promise<GPUBuffer | null> {
//     if (!device) {
//         console.error('WebGPU device not initialized.');
//         return null;
//     }

//     // Dimensions
//     const cubeSize = 256;
//     const xNorm = stylePosGuide.width / cubeSize;
//     const yNorm = stylePosGuide.height / cubeSize;

//     // Compute shader
//     const computeShaderCode = `
//         @group(0) @binding(0) var<storage, read> posGuide: array<u32>;
//         @group(0) @binding(1) var<storage, read> appGuide: array<u32>;
//         @group(0) @binding(2) var<storage, read_write> lookUpCube: array<u32>;

//         @compute @workgroup_size(1, 1, 1)
//         fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//         let x = id.x;
//         let y = id.y;
//         let z = id.z;

//         let seedRow = u32(f32(y) * ${yNorm});
//         let seedCol = u32(f32(x) * ${xNorm});
//         var minError: u32 = 0xFFFFFFFF;
//         var bestCoord: vec2<u32> = vec2<u32>(0, 0);

//         for (var rowOffset: i32 = -${radius}; rowOffset <= ${radius}; rowOffset++) {
//             let row = i32(seedRow) + rowOffset;
//             if (row < 0 || row >= ${stylePosGuide.height}) {
//             continue;
//             }
//             for (var colOffset: i32 = -${radius}; colOffset <= ${radius}; colOffset++) {
//             let col = i32(seedCol) + colOffset;
//             if (col < 0 || col >= ${stylePosGuide.width}) {
//                 continue;
//             }
//             let error = abs(posGuide[u32(row) * ${stylePosGuide.width} + u32(col)] - z);
//             if (error < minError) {
//                 minError = error;
//                 bestCoord = vec2<u32>(u32(col), u32(row));
//             }
//             }
//         }

//         let index = x * ${cubeSize} * ${cubeSize} + y * ${cubeSize} + z;
//         lookUpCube[index * 2] = bestCoord.x;
//         lookUpCube[index * 2 + 1] = bestCoord.y;
//         }
//     `;

//     // Create GPU resources
//     const shaderModule = device.createShaderModule({ code: computeShaderCode });
//     const bindGroupLayout = device.createBindGroupLayout({
//         entries: [
//         { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
//         { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
//         { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
//         ],
//     });

//     const pipeline = device.createComputePipeline({
//         layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
//         compute: { module: shaderModule, entryPoint: 'main' },
//     });

//     const posGuideBuffer = device.createBuffer({
//         size: stylePosGuide.data.length * Uint32Array.BYTES_PER_ELEMENT,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//         mappedAtCreation: true,
//     });
//     new Uint32Array(posGuideBuffer.getMappedRange()).set(new Uint32Array(stylePosGuide.data.buffer));
//     posGuideBuffer.unmap();

//     const appGuideBuffer = device.createBuffer({
//         size: styleAppGuide.data.length * Uint32Array.BYTES_PER_ELEMENT,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//         mappedAtCreation: true,
//     });
//     new Uint32Array(appGuideBuffer.getMappedRange()).set(new Uint32Array(styleAppGuide.data.buffer));
//     appGuideBuffer.unmap();

//     const lookUpCubeBuffer = device.createBuffer({
//         size: cubeSize * cubeSize * cubeSize * 2 * Uint32Array.BYTES_PER_ELEMENT,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
//     });

//     const stagingBuffer = device.createBuffer({
//         size: cubeSize * cubeSize * cubeSize * 2 * Uint32Array.BYTES_PER_ELEMENT,
//         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
//     });
      

//     // Bind group
//     const bindGroup = device.createBindGroup({
//         layout: bindGroupLayout,
//         entries: [
//         { binding: 0, resource: { buffer: posGuideBuffer } },
//         { binding: 1, resource: { buffer: appGuideBuffer } },
//         { binding: 2, resource: { buffer: lookUpCubeBuffer } },
//         ],
//     });

//     // Command encoder
//     const commandEncoder = device.createCommandEncoder();
//     const passEncoder = commandEncoder.beginComputePass();
//     passEncoder.setPipeline(pipeline);
//     passEncoder.setBindGroup(0, bindGroup);
//     passEncoder.dispatchWorkgroups(cubeSize / 8, cubeSize / 8, cubeSize / 8);
//     passEncoder.end();

//     commandEncoder.copyBufferToBuffer(lookUpCubeBuffer, 0, stagingBuffer, 0, stagingBuffer.size);

//     // Submit commands
//     device.queue.submit([commandEncoder.finish()]);

//     // Wait for completion
//     await device.queue.onSubmittedWorkDone();

//     return stagingBuffer;
//   }

}