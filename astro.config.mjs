// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://shokuno5.github.io',
  // No base path needed for username.github.io sites
  integrations: [mdx(), tailwind()]
});