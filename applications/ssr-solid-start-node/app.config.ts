import { defineConfig } from "@solidjs/start/config";

export default defineConfig({
  server: {
    nitro: {
      compressPublicAssets: false
    }
  }
});
