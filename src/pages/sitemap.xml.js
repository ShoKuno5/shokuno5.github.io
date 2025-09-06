import { getCollection } from 'astro:content';

export async function GET({ site }) {
  const urls = [];

  // Static core pages
  const staticPaths = [
    '/',
    '/about/',
    '/projects/',
    '/persona/',
    '/posts/all/',
    '/posts/tags/',
    '/research/',
    '/media/',
    '/ja/',
    '/ja/about/',
    '/ja/projects/',
    '/ja/posts/all/',
    '/ja/posts/tags/',
    '/ja/research/',
    '/ja/media/',
  ];

  for (const p of staticPaths) {
    urls.push({ loc: new URL(p, site).toString(), changefreq: 'weekly', priority: 0.6 });
  }

  // Blog posts (en + ja)
  const posts = await getCollection('posts');
  for (const post of posts) {
    // Normalize path: strip language prefix for en, keep ja under /ja/
    const isJa = post.slug.startsWith('ja/');
    const slug = isJa ? post.slug.slice(3) : post.slug.replace(/^en\//, '');
    const path = isJa ? `/ja/posts/${slug}/` : `/posts/${slug}/`;
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

