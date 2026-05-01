import { defineConfig } from 'vite'
import { devtools } from '@tanstack/devtools-vite'
import { tanstackStart } from '@tanstack/solid-start/plugin/vite'
import viteSolid from 'vite-plugin-solid'
import { nitro } from 'nitro/vite'

const config = defineConfig({
  plugins: [
    devtools(),
    tanstackStart(),
    nitro({
      compressPublicAssets: false
    }),
    // solid's vite plugin must come after start's vite plugin
    viteSolid({ ssr: true }),
  ],
})

export default config
