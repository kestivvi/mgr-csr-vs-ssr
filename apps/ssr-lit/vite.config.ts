import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    outDir: 'dist/static',
    emptyOutDir: false,
    lib: {
      entry: path.resolve(__dirname, 'src/client.ts'),
      name: 'client',
      fileName: 'client',
      formats: ['es'],
    },
    rollupOptions: {
      output: {
        entryFileNames: 'client.js',
      },
    },
  },
});
