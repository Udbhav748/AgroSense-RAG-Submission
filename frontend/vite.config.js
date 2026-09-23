import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
    globals: true,
  },
  server: {
    proxy: {
      // api.js/chatService.js call the backend through a same-origin /api
      // base URL in dev; this proxies those calls to FastAPI (stripping
      // the /api prefix, since the backend's own routes live at /upload,
      // /chat, etc.). Keeping the browser on a single origin means no CORS
      // and no dependency on the browser being able to reach the backend
      // host directly — works from localhost and LAN IPs alike.
      '/api': {
        target: 'http://localhost:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
