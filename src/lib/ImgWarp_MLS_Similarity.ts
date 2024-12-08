export interface Point {
    x: number;
    y: number;
}

export class ImgWarp_MLS_Similarity {
    alpha: number = 1.0;
    gridSize: number = 5;

    oldDotL: Point[] = [];
    newDotL: Point[] = [];
    nPoint: number = 0;

    rDx: number[][] = [];
    rDy: number[][] = [];

    srcW: number = 0;
    srcH: number = 0;
    tarW: number = 0;
    tarH: number = 0;

    setAllAndGenerate(
        srcImageData: ImageData,
        qsrc: Point[],
        qdst: Point[],
        outW: number,
        outH: number,
        transRatio: number = 1
    ): ImageData {
        this.setSize(srcImageData.width, srcImageData.height);
        this.setTargetSize(outW, outH);
        this.setSrcPoints(qsrc);
        this.setDstPoints(qdst);
        this.calcDelta();
        return this.genNewImageData(srcImageData, transRatio);
    }

    setSize(w: number, h: number) {
        this.srcW = w;
        this.srcH = h;
    }

    setTargetSize(outW: number, outH: number) {
        this.tarW = outW;
        this.tarH = outH;
    }

    setSrcPoints(qsrc: Point[]) {
        this.nPoint = qsrc.length;
        this.newDotL = qsrc.map(point => ({ ...point }));
    }

    setDstPoints(qdst: Point[]) {
        this.nPoint = qdst.length;
        this.oldDotL = qdst.map(point => ({ ...point }));
    }

    calcDelta() {
        const { tarW, tarH, gridSize, oldDotL, newDotL, nPoint } = this;

        this.rDx = Array.from({ length: tarH }, () => Array(tarW).fill(0));
        this.rDy = Array.from({ length: tarH }, () => Array(tarW).fill(0));

        if (nPoint < 2) return;

        const w = new Array(nPoint);

        for (let i = 0; i < tarW; i += gridSize) {
            for (let j = 0; j < tarH; j += gridSize) {
                let sw = 0;
                let swp = { x: 0, y: 0 };
                let swq = { x: 0, y: 0 };
                let newP = { x: 0, y: 0 };
                const curV = { x: i, y: j };

                let k = 0;
                for (; k < nPoint; k++) {
                    if (i === oldDotL[k].x && j === oldDotL[k].y) break;
                    const dx = i - oldDotL[k].x;
                    const dy = j - oldDotL[k].y;
                    const dist2 = dx * dx + dy * dy;
                    w[k] = 1 / dist2;

                    sw += w[k];
                    swp.x += w[k] * oldDotL[k].x;
                    swp.y += w[k] * oldDotL[k].y;
                    swq.x += w[k] * newDotL[k].x;
                    swq.y += w[k] * newDotL[k].y;
                }

                if (k === nPoint) {
                    const pstar = { x: swp.x / sw, y: swp.y / sw };
                    const qstar = { x: swq.x / sw, y: swq.y / sw };

                    let miu_s = 0;
                    for (k = 0; k < nPoint; k++) {
                        const Pi = {
                            x: oldDotL[k].x - pstar.x,
                            y: oldDotL[k].y - pstar.y,
                        };
                        miu_s += w[k] * (Pi.x * Pi.x + Pi.y * Pi.y);
                    }

                    const curVStar = {
                        x: curV.x - pstar.x,
                        y: curV.y - pstar.y,
                    };
                    const curVJ = { x: -curVStar.y, y: curVStar.x };

                    newP = { x: 0, y: 0 };
                    for (k = 0; k < nPoint; k++) {
                        const Pi = {
                            x: oldDotL[k].x - pstar.x,
                            y: oldDotL[k].y - pstar.y,
                        };
                        const PiJ = { x: -Pi.y, y: Pi.x };

                        const a = (Pi.x * curVStar.x + Pi.y * curVStar.y) * newDotL[k].x -
                                (PiJ.x * curVStar.x + PiJ.y * curVStar.y) * newDotL[k].y;
                        const b = -(Pi.x * curVJ.x + Pi.y * curVJ.y) * newDotL[k].x +
                                    (PiJ.x * curVJ.x + PiJ.y * curVJ.y) * newDotL[k].y;

                        const coeff = w[k] / miu_s;
                        newP.x += coeff * a;
                        newP.y += coeff * b;
                    }

                    newP.x += qstar.x;
                    newP.y += qstar.y;
                } else {
                    newP = { x: newDotL[k].x, y: newDotL[k].y };
                }

                this.rDx[j][i] = newP.x - i;
                this.rDy[j][i] = newP.y - j;
            }
        }
    }

    genNewImageData(srcImageData: ImageData, transRatio: number): ImageData {
        const { tarW, tarH, gridSize, rDx, rDy, srcW, srcH } = this;
        const oriData = srcImageData.data;

        const newImageData = new ImageData(tarW, tarH);
        const newData = newImageData.data;

        for (let i = 0; i < tarH; i += gridSize) {
            for (let j = 0; j < tarW; j += gridSize) {
                const ni = Math.min(i + gridSize, tarH - 1);
                const nj = Math.min(j + gridSize, tarW - 1);
                const h = ni - i;
                const w = nj - j;

                for (let di = 0; di <= h; di++) {
                    for (let dj = 0; dj <= w; dj++) {
                        const deltaX = bilinearInterp(
                            di / h,
                            dj / w,
                            rDx[i][j],
                            rDx[i][nj],
                            rDx[ni][j],
                            rDx[ni][nj]
                        ) * transRatio;
                        const deltaY = bilinearInterp(
                            di / h,
                            dj / w,
                            rDy[i][j],
                            rDy[i][nj],
                            rDy[ni][j],
                            rDy[ni][nj]
                        ) * transRatio;

                        let nx = j + dj + deltaX;
                        let ny = i + di + deltaY;

                        nx = Math.max(0, Math.min(srcW - 1, nx));
                        ny = Math.max(0, Math.min(srcH - 1, ny));

                        const nxi = Math.floor(nx);
                        const nyi = Math.floor(ny);
                        const nxi1 = Math.min(nxi + 1, srcW - 1);
                        const nyi1 = Math.min(nyi + 1, srcH - 1);

                        const dx = nx - nxi;
                        const dy = ny - nyi;

                        const idxSrc = (nyi * srcW + nxi) * 4;
                        const idxSrcX = (nyi * srcW + nxi1) * 4;
                        const idxSrcY = (nyi1 * srcW + nxi) * 4;
                        const idxSrcXY = (nyi1 * srcW + nxi1) * 4;
                        const idxDst = ((i + di) * tarW + (j + dj)) * 4;

                        for (let k = 0; k < 4; k++) {
                            const val = bilinearInterp(
                                dx,
                                dy,
                                oriData[idxSrc + k],
                                oriData[idxSrcX + k],
                                oriData[idxSrcY + k],
                                oriData[idxSrcXY + k]
                            );
                            newData[idxDst + k] = val;
                        }
                    }
                }
            }
        }

        return newImageData;
    }
}

function bilinearInterp(x: number, y: number, v00: number, v01: number, v10: number, v11: number): number {
    return (1 - x) * (1 - y) * v00 +
            x * (1 - y) * v01 +
            (1 - x) * y * v10 +
            x * y * v11;
}
