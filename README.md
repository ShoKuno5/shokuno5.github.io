# Myfolio

Minimal personal research and writing site built with Astro. The site is a single stream of posts on the home page, with each post available on its own URL.

## Structure
```
src/
├── content/posts/          # Markdown posts
├── layouts/Layout.astro    # Global shell
├── layouts/PostLayout.astro# Post page layout
├── pages/index.astro       # Home feed
└── pages/posts/[...slug].astro # Post pages
```

## Development
```bash
npm install
npm run dev
npm run check
npm run build
```

Math and citations are supported via KaTeX + the existing bibliography utilities.
