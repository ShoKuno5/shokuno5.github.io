# Astro Blog with LaTeX Math & BibTeX Citations

Academic blog setup with build-time math rendering and automatic citation processing.

## Quick Start

```bash
bash setup.sh
npm run dev
```

## Features

- **LaTeX Math**: Inline `$...$` and display `$$...$$` equations via KaTeX
- **BibTeX Citations**: Pandoc-style `[@key]` citations with Vancouver numbering
- **Auto Bibliography**: References section generated automatically

## Usage

1. Add citations to `src/content/refs/library.bib`
2. Write posts with math: `$E = mc^2$` and citations: `[@authorYear]`
3. Build-time processing converts to numbered references

## Files Created

- `astro.config.mjs` - remark-math, rehype-katex, rehype-citation config
- `src/layouts/BaseLayout.astro` - KaTeX CSS from CDN
- `src/content/refs/library.bib` - Sample BibTeX entries
- `src/pages/posts/sample-post.md` - Demo with math & citations