// Builds the interface into the Python package, so FastAPI serves it and team
// members never need Node. `npm run dev` proxies /api to a running launcher.
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "../src/foxtrot/api/static",
    emptyOutDir: true,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
