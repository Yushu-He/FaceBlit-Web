import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { openDB } from 'idb';
import type { IDBPDatabase } from 'idb';

export interface StyleData {
  styleName: string;
  data: {
    resizedImage: ImageData,
    gradient: ImageData,
    appGuide: ImageData,
    lookupCube: Uint16Array,
    landmarks: NormalizedLandmark[][],
  }
}

export class DB {
  private dbPromise: Promise<IDBPDatabase<any>>;

  constructor() {
    this.dbPromise = openDB('styleDatabase', 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('styles')) {
          db.createObjectStore('styles', { keyPath: 'styleName' });
        }
      },
    });
    
  }

  async saveStyleData(styleName: string, data: any) {
    const db = await this.dbPromise;
    await db.put('styles', { styleName, ...data });
  }

  async loadStyleData(styleName: string) {
    const db = await this.dbPromise;
    try {
      return await db.get('styles', styleName);
    } catch (e) {
      return null;
    }

  }
}
