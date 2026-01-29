// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { rehypeLazyImages } from './src/utils/rehype-lazy-images.js';
import { rehypeCitations } from './src/utils/rehype-citations.js';

const CITATIONS_VERSION = 9;

export default defineConfig({
  site: 'https://shokuno5.github.io',
  output: 'static',
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        rehypeLazyImages,
        [rehypeCitations, { sectionTitle: 'References', version: CITATIONS_VERSION }]
      ]
    }),
    tailwind()
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      rehypeLazyImages,
      [rehypeCitations, { sectionTitle: 'References', version: CITATIONS_VERSION }]
    ],
    syntaxHighlight: 'shiki',
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      }
    }
  }
});
