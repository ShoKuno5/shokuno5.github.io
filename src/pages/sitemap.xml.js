import { getCollection } from 'astro:content';
import { STATIC_PATHS } from '../config/site.js';

export async function GET({ site }) {
  const urls = [];

  // Static core pages
  const staticPaths = STATIC_PATHS;

  for (const p of staticPaths) {
    urls.push({ loc: new URL(p, site).toString(), changefreq: 'weekly', priority: 0.6 });
  }

  // Blog posts (English only)
  const posts = await getCollection('posts');
  for (const post of posts) {
    if (post.slug.startsWith('ja/')) continue; // skip Japanese
    const slug = post.slug.replace(/^en\//, '');
    const path = `/posts/${slug}/`;
    urls.push({ loc: new URL(path, site).toString(), changefreq: 'monthly', priority: 0.7 });
  }

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
  <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    ${urls
      .map(
        (u) => `<url><loc>${u.loc}</loc><changefreq>${u.changefreq}</changefreq><priority>${u.priority}</priority></url>`
      )
      .join('')}
  </urlset>`;

  return new Response(xml, {
    status: 200,
    headers: { 'Content-Type': 'application/xml' },
  });
}
