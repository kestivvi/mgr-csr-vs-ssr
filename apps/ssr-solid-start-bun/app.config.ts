import { defineConfig } from "@solidjs/start/config";

export default defineConfig({
  server: {
    nitro: {
      preset: 'bun',
      compressPublicAssets: false
    }
  }
});
