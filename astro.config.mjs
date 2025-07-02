// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeCitation from 'rehype-citation';

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
        rehypeKatex,
        [rehypeCitation, {
          bibliography: './src/content/refs/library.bib',
          csl: 'vancouver',
          linkCitations: true
        }]
      ]
    }),
    tailwind()
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypeCitation, {
        bibliography: './src/content/refs/library.bib',
        csl: 'vancouver',
        linkCitations: true
      }]
    ]
  }
});