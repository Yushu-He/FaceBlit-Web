import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  optimizeDeps: {
    include: ['@mediapipe/tasks-vision']
  },
  build: {
    outDir: 'build', // 指定输出目录
    emptyOutDir: true,
  },
  base: './', // 设置相对路径，适合 GitHub Pages 部署
});