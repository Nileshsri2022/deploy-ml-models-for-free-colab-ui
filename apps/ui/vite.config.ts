import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI talks to the gateway with *relative* URLs only. In development Vite
// proxies /api, /v1 and /internal to the gateway, so the browser never needs to
// know where the backend lives - which is also what makes the sandbox preview
// (https://{port}-{sandbox}.e2b.app) work without CORS or hardcoded hosts.
const target = process.env.CDP_GATEWAY_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: false,
    allowedHosts: true,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/v1': { target, changeOrigin: true },
      '/internal': { target, changeOrigin: true },
      '/healthz': { target, changeOrigin: true },
      '/docs': { target, changeOrigin: true },
      '/openapi.json': { target, changeOrigin: true },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/v1': { target, changeOrigin: true },
      '/internal': { target, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 900,
  },
})
