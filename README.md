# Myfolio

Personal research and writing site for Sho Kuno, built with Astro and a content-first workflow. The project favors simple markdown sources, minimal client JavaScript, and a tidy build output so the site can be deployed as static files.

## Architecture
- **Astro 5 + MDX** for static rendering and math support via KaTeX
- **Tailwind CSS** for utility styling (global styles live in `src/styles/`)
- **Content collections** defined in `src/content/config.ts` to keep markdown frontmatter strongly typed
- **Single-language** navigation assembled from `src/config/site.js`

### Content layout
```
src/content/
├── face/face.md           # Home page body copy
├── research/research.md   # "Salients" research page
├── about/about.md         # About page copy
├── posts/en/*.md          # Blog posts (English)
└── projects-page/projects.md  # Card data for /projects/
```

Static assets such as the downloadable CV live in `public/` (e.g. `/updated_cv.pdf`). Navigation items read from `src/config/site.js`, while page shells live in `src/pages/*.astro`.

## Development
```bash
npm install
npm run dev
npm run build
npm run preview
```

The build is fully static and inlines global CSS for faster first paint. A prebuild script (`scripts/prebuild.js`) caches git dates for blog posts.

## Notes
- Removing unused files: legacy CV artifacts have been pruned so `/updated_cv.pdf` is the single source
- Markdown-first: always add content through the collections above rather than hardcoding copy in `.astro` files
- Layout components (`src/layouts/`) provide shared markup, side navigation, and meta tags

This repository is intentionally concise—keep new dependencies minimal and continue to organize content through the collections.
