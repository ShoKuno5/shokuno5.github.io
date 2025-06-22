# Myfolio

A minimalist, single-column portfolio site built with Astro, MDX, and Tailwind CSS.

## Features

- 🎨 Clean, Sam Altman-inspired single-column layout
- 📝 MDX support for rich content
- 🎯 Tailwind CSS for styling
- 📱 Fully responsive design
- 🚀 GitHub Pages deployment ready
- ⚡ Fast static site generation

## Getting Started

### Development

```bash
npm install
npm run dev
```

Visit `http://localhost:4321` to see your site.

### Building

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Structure

- `/src/pages/` - Main pages (About, Projects, Blog, Research)
- `/src/content/` - Markdown/MDX content
  - `/blog/` - Blog posts
  - `/projects/` - Project descriptions
- `/src/layouts/` - Page layouts
- `/src/styles/` - Global styles

## Customization

### Accent Color
Change the accent color in `/src/styles/global.css`:
```css
:root {
  --accent: #007aff; /* Change this */
}
```

### Hero Text
Edit the hero section in `/src/pages/index.astro`

### Navigation
Update navigation links in `/src/layouts/Layout.astro`

## Deployment

### GitHub Pages

1. Update `astro.config.mjs`:
   - Set `site` to your GitHub Pages URL
   - Set `base` to your repository name

2. Push to GitHub

3. Enable GitHub Pages in repository settings

4. The site will automatically deploy on push to main branch

### Custom Domain

1. Add a CNAME file to `/public/` with your domain
2. Configure DNS with your provider (Cloudflare recommended)

## Adding Content

### Blog Posts
Create `.md` or `.mdx` files in `/src/content/blog/`:

```markdown
---
title: "Your Post Title"
description: "Post description"
date: 2024-03-20
tags: ["tag1", "tag2"]
---

Your content here...
```

### Projects
Create files in `/src/content/projects/`:

```markdown
---
title: "Project Name"
description: "Project description"
tags: ["Tech", "Stack"]
github: "https://github.com/..."
featured: true
---

Project details...
```
