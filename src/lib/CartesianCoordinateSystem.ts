import type { Point } from './ImgWarp_MLS_Similarity';
  
export class CartesianCoordinateSystem {
    getAveragePoint(points: Point[]): Point {
        let sumX = 0;
        let sumY = 0;
        for (const point of points) {
        sumX += point.x;
        sumY += point.y;
        }
        return {
        x: sumX / points.length,
        y: sumY / points.length,
        };
    }

    averageMarkers(A: Point[], B: Point[]): Point[] {
        return A.map((pointA, i) => ({
        x: (pointA.x + B[i].x) / 2,
        y: (pointA.y + B[i].y) / 2,
        }));
    }

    static isPointInsideImage(point: Point, image: ImageData): boolean {
        return (
        point.x >= 0 &&
        point.y >= 0 &&
        point.x < image.width &&
        point.y < image.height
        );
    }

    clampPointInsideImage(point: Point, size: { width: number; height: number }) {
        point.x = Math.max(0, Math.min(point.x, size.width - 1));
        point.y = Math.max(0, Math.min(point.y, size.height - 1));
    }

    getPointLyingOnCircle(
        center: Point,
        radius: number,
        theta: number,
        size: { width: number; height: number } = { width: 768, height: 1024 }
    ): Point {
        const rad = (theta * Math.PI) / 180;
        const x = center.x + radius * Math.cos(rad);
        const y = center.y + radius * Math.sin(rad);
        const point = { x, y };
        this.clampPointInsideImage(point, size);
        return point;
    }

    recomputePointsDueToScale(
        points: Point[],
        scaleRatio: number,
        origin: Point,
        dstSize: { width: number; height: number }
    ): Point[] {
        return points.map((point) => {
        const x = Math.round((point.x - origin.x) * scaleRatio + origin.x);
        const y = Math.round((point.y - origin.y) * scaleRatio + origin.y);
        const newPoint = { x, y };
        this.clampPointInsideImage(newPoint, dstSize);
        return newPoint;
        });
    }

    recomputePointsDueToTranslation(
        points: Point[],
        shift: Point,
        size: { width: number; height: number }
    ): Point[] {
        return points.map((point) => {
        const newPoint = { x: point.x + shift.x, y: point.y + shift.y };
        this.clampPointInsideImage(newPoint, size);
        return newPoint;
        });
    }

    recomputePoints180Rotation(
        points: Point[],
        size: { width: number; height: number }
    ): Point[] {
        const resultPoints: Point[] = new Array(points.length);
        for (let i = 0; i < points.length; i++) {
        const newX = size.width - points[i].x;
        const newY = size.height - points[i].y;
        let index = i;
        if (points.length === 68) {
            if (i >= 0 && i < 17) index = 16 - i;
            else if (i >= 17 && i < 27) index = 43 - i;
            else if (i >= 27 && i < 31) index = i;
            else if (i >= 31 && i < 36) index = 66 - i;
            else if ((i >= 36 && i < 40) || (i >= 42 && i < 46)) index = 81 - i;
            else if ((i >= 40 && i < 42) || (i >= 46 && i < 48)) index = 87 - i;
            else if (i >= 48 && i < 55) index = 102 - i;
            else if (i >= 55 && i < 60) index = 114 - i;
            else if (i >= 60 && i < 65) index = 124 - i;
            else index = 132 - i;
        }
        resultPoints[index] = { x: newX, y: newY };
        }
        return resultPoints;
    }

    getHeadAreaRect(
        faceLandmarks: Point[],
        imgSize: { width: number; height: number }
    ): { x: number; y: number; width: number; height: number } {
        const width = faceLandmarks[16].x - faceLandmarks[0].x;
        const higherY = Math.min(faceLandmarks[0].y, faceLandmarks[16].y);
        const height = faceLandmarks[8].y - (higherY - width / 2);
        const rectX = Math.max(faceLandmarks[0].x - width * 0.1, 0);
        const rectY = Math.max(higherY - width / 2 - height * 0.2, 0);
        const rectWidth = Math.min(width * 1.2, imgSize.width - rectX);
        const rectHeight = Math.min(height * 1.4, imgSize.height - rectY);
        return {
        x: rectX,
        y: rectY,
        width: rectWidth,
        height: rectHeight,
        };
    }

    euclideanDistance(p1: Point, p2: Point): number {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    getDistancesOfEachPointFromOrigin(points: Point[]): { point: Point; distance: number }[] {
        let cumDistance = 0;
        const result: { point: Point; distance: number }[] = [];
        for (let i = 0; i < points.length; i++) {
        result.push({ point: points[i], distance: cumDistance });
        if (i + 1 < points.length) {
            cumDistance += this.euclideanDistance(points[i], points[i + 1]);
        }
        }
        return result;
    }

    interpolate(A: Point, B: Point, distBI: number): Point {
        const distAB = this.euclideanDistance(A, B);
        const distAI = distAB - distBI;
        const ratio = distAI / distAB;
        const xI = Math.round(A.x + ratio * (B.x - A.x));
        const yI = Math.round(A.y + ratio * (B.y - A.y));
        return { x: xI, y: yI };
    }

    getContourPointsSubset(points: Point[], subsetSize: number): Point[] {
        points.push(points[0]); // 闭合轮廓
        const pointsAndDistances = this.getDistancesOfEachPointFromOrigin(points);
        const newPoints: Point[] = [];
        const totalDistance = pointsAndDistances[pointsAndDistances.length - 1].distance;
        const distanceBetweenNewPoints = totalDistance / subsetSize;
        let currDistance = 0;
        let oldIndex = 0;
        let oldA = pointsAndDistances[oldIndex];
        let oldB = pointsAndDistances[oldIndex + 1];

        for (let i = 0; i < subsetSize; i++) {
        currDistance = distanceBetweenNewPoints * i;
        while (currDistance < oldA.distance || currDistance > oldB.distance) {
            oldA = pointsAndDistances[++oldIndex];
            oldB = pointsAndDistances[oldIndex + 1];
        }
        const distBI = oldB.distance - currDistance;
        newPoints.push(this.interpolate(oldA.point, oldB.point, distBI));
        }

        return newPoints;
    }

    drawLandmarks(
        ctx: CanvasRenderingContext2D,
        landmarks: Point[],
        color: string = 'green',
        shape: 'circles' | 'lines' = 'circles'
    ) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        if (shape === 'circles') {
        for (const point of landmarks) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
        } else if (shape === 'lines') {
        ctx.beginPath();
        ctx.moveTo(landmarks[0].x, landmarks[0].y);
        for (let i = 1; i < landmarks.length; i++) {
            if ([16, 21, 26, 35, 41, 47, 59].includes(i)) {
            ctx.moveTo(landmarks[i].x, landmarks[i].y);
            } else {
            ctx.lineTo(landmarks[i].x, landmarks[i].y);
            }
        }
        ctx.stroke();
        }
    }

    savePointsIntoFile(points: Point[], filename: string) {
        const data = points.map((point) => `${point.x} ${point.y}`).join('\n');
        const blob = new Blob([data], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}