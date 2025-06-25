# Bilingual Blog Architecture

## Complete Language Support
This blog is fully bilingual with complete English and Japanese versions of ALL pages.

## Structure
```
src/
├── content/posts/
│   ├── en/          # English posts
│   └── ja/          # Japanese posts
├── pages/
│   ├── *.astro      # English pages
│   └── ja/
│       └── *.astro  # Japanese pages
```

## URLs
- **English** (default): `/`, `/posts/all/`, `/about/`, `/projects/`, etc.
- **Japanese**: `/ja/`, `/ja/posts/all/`, `/ja/about/`, `/ja/projects/`, etc.

## Navigation
- Navigation is completely language-aware
- When on Japanese pages, ALL links point to Japanese versions
- When on English pages, ALL links point to English versions
- Language switcher: subtle "en/ja" in top navigation

## Translation Workflow

### For Blog Posts
```bash
node translate.js <filename>
```

Example:
```bash
node translate.js 2023-08-15-python-for-data-science.md
```

The script will:
1. Display English content
2. Provide Claude prompt
3. Accept Japanese translation input
4. Save to correct `ja/` directory

### Claude Translation Prompt
"Translate this markdown blog post to Japanese, keeping all frontmatter and code blocks intact:"

### For Static Pages
Japanese versions already exist for all static pages:
- `/ja/about.astro`
- `/ja/projects.astro` 
- `/ja/naive-hope.astro`
- `/ja/posts/all.astro`
- `/ja/posts/tags.astro`

## Technical Details
- Language detection: automatic based on URL (`/ja/` prefix)
- Navigation: dynamically generated per language
- Git dates: fallback to original file paths for history
- Modified dates: working correctly on all pages

## Build/Test Commands
- Development: `npm run dev`
- Build: `npm run build`
- Preview: `npm run preview`