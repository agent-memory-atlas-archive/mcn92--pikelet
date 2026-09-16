import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
    root: resolve(__dirname),
    base: './',
    build: {
        target: 'es2022',
        outDir: resolve(__dirname, '..', 'playground-dist'),
        emptyOutDir: true,
    },
});
