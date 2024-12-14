# Face Blit Web

This project brings [Face Blit](https://github.com/AnetaTexler/FaceBlit) to a web application for demonstration purposes.

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Yushu-He/FaceBlit-Web.git
    cd FaceBlit-Web
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Run the development server:**

    ```bash
    npm run dev
    ```

## Technologies Used

- [Svelte](https://svelte.dev/)
- [WebAssembly (Wasm)](https://webassembly.org/)
- [MediaPipe](https://github.com/google/mediapipe)
- [IDB](https://github.com/jakearchibald/idb)
- [Font Awesome](https://fontawesome.com/)
- [Vite](https://vitejs.dev/)

## Note

In the main branch, there is a TypeScript native version, but due to time constraints, it is only partially completed and has low performance.

This project uses IndexedDB to cache the large lookup table after loading to improve page loading speed. However, due to Apple's limitations on the storage space of IndexedDB, this feature does not work on the WebKit engine.

## Conclusion

This project was developed as the final project for the University of Michigan EECS442 course.

## References

- [Face Blit](https://github.com/AnetaTexler/FaceBlit)
- [MediaPipe](https://github.com/google/mediapipe)
- [Mediapipe 2 Dlib Landmarks by PeizhiYan](https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks/tree/main)
- [Build TFLite & OpenCV to WASM (with SIMD, CMake) | Face Detection on Web](https://blog.seeso.io/face-detection-on-web-tflite-wasm-simd-462975e0f628)
- University of Michigan EECS442 Course Materials