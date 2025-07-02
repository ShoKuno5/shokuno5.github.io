// @ts-check
import { defineConfig } from 'astro/config';
import remarkCite from '@benrbray/remark-cite';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

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
  markdown: {
    remarkPlugins: [
      [remarkCite, {
        bibliography: 'src/content/bibliography.bib'
      }]
    ]
  },
  integrations: [
    mdx({
      remarkPlugins: [
        [remarkCite, {
          bibliography: 'src/content/bibliography.bib'
        }]
      ]
    }),
    tailwind()
  ]
});