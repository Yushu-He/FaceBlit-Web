// credit: https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks/tree/main
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

const mp2dlib_correspondence = [
    // Face Contour
    [127],       // 1
    [234],       // 2
    [93],        // 3
    [132, 58],   // 4
    [58, 172],   // 5
    [136],       // 6
    [150],       // 7
    [176],       // 8
    [152],       // 9
    [400],       // 10
    [379],       // 11
    [365],       // 12
    [397, 288],  // 13
    [361],       // 14
    [323],       // 15
    [454],       // 16
    [356],       // 17

    // Right Brow
    [70],        // 18
    [63],        // 19
    [105],       // 20
    [66],        // 21
    [107],       // 22

    // Left Brow
    [336],       // 23
    [296],       // 24
    [334],       // 25
    [293],       // 26
    [300],       // 27

    // Nose
    [168, 6],    // 28
    [197, 195],  // 29
    [5],         // 30
    [4],         // 31
    [75],        // 32
    [97],        // 33
    [2],         // 34
    [326],       // 35
    [305],       // 36

    // Right Eye
    [33],        // 37
    [160],       // 38
    [158],       // 39
    [133],       // 40
    [153],       // 41
    [144],       // 42

    // Left Eye
    [362],       // 43
    [385],       // 44
    [387],       // 45
    [263],       // 46
    [373],       // 47
    [380],       // 48

    // Upper Lip Contour Top
    [61],        // 49
    [39],        // 50
    [37],        // 51
    [0],         // 52
    [267],       // 53
    [269],       // 54
    [291],       // 55

    // Lower Lip Contour Bottom
    [321],       // 56
    [314],       // 57
    [17],        // 58
    [84],        // 59
    [91],        // 60

    // Upper Lip Contour Bottom
    [78],        // 61
    [82],        // 62
    [13],        // 63
    [312],       // 64
    [308],       // 65

    // Lower Lip Contour Top
    [317],       // 66
    [14],        // 67
    [87],        // 68
  ];

export class MP2Dlib {
  constructor() {
    for (let i = 0; i < mp2dlib_correspondence.length; i++) {
        if (mp2dlib_correspondence[i].length === 1) {
          const idx = mp2dlib_correspondence[i][0];
          mp2dlib_correspondence[i] = [idx, idx];
        }
    }
  }

  transformLandmarks(landmarks: NormalizedLandmark[][]): NormalizedLandmark[][] {
    const landmarks_extracted: { x: number; y: number; z: number; visibility: number }[][] = [];
    for (const landmark of landmarks) {
      const landmark_extracted: { x: number; y: number; z: number; visibility: number }[] = [];
      for (const indices of mp2dlib_correspondence) {
        let x = 0;
        let y = 0;
        let z = 0;

        for (const idx of indices) {
          x += landmark[idx].x;
          y += landmark[idx].y;
          z += landmark[idx].z || 0;
        }

        const numPoints = indices.length;
        landmark_extracted.push({
          x: x / numPoints,
          y: y / numPoints,
          z: z / numPoints,
          visibility: 1,
        });
      }
    landmarks_extracted.push(landmark_extracted);
    }
    return landmarks_extracted;
  }
}