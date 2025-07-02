// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// import rehypeCitation from 'rehype-citation'; // Disabled - using client-side citations

// https://astro.build/config
export default defineConfig({
  site: 'https://shokuno5.github.io',
  // No base path needed for username.github.io sites
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ja'],
    routing: {
      prefixDefaultLocale: false
    }
  },
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex
        // [rehypeCitation, { ... }] // Disabled - using client-side citations
      ]
    }),
    tailwind()
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex
      // [rehypeCitation, { ... }] // Disabled - using client-side citations
    ]
  }
});