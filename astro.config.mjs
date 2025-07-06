// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { rehypeLazyImages } from './src/utils/rehype-lazy-images.js';

// https://astro.build/config
export default defineConfig({
  site: 'https://shokuno5.github.io/PrimalTimes.github.io',
  output: 'static',
  compressHTML: true,
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ja'],
    routing: {
      prefixDefaultLocale: false
    }
  },
  build: {
    assets: 'assets',
    inlineStylesheets: 'auto',
  },
  vite: {
    build: {
      cssCodeSplit: false, // Inline all CSS
      rollupOptions: {
        output: {
          manualChunks: undefined,
          assetFileNames: 'assets/[name].[hash][extname]',
          chunkFileNames: 'assets/[name].[hash].js',
          entryFileNames: 'assets/[name].[hash].js'
        }
      },
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true, // Remove console.logs in production
          drop_debugger: true,
          pure_funcs: ['console.log', 'console.info', 'console.debug']
        }
      }
    },
    ssr: {
      noExternal: ['@astrojs/mdx']
    }
  },
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        rehypeLazyImages
        // [rehypeCitation, { ... }] // Disabled - using client-side citations
      ]
    }),
    tailwind()
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      rehypeLazyImages
      // [rehypeCitation, { ... }] // Disabled - using client-side citations
    ]
  }
});