# Citation System Usage Guide

This blog uses a **unified client-side citation system** that automatically processes `\cite{}` commands and generates numbered references with APA 7th edition formatting.

## Quick Start

1. **Write clean Markdown**: Use standard Markdown syntax with LaTeX math (`$...$`, `$$...$$`)
2. **Add citations**: Use `\cite{key}` where `key` matches an entry in your bibliography file
3. **No layout needed**: Posts automatically use the citation system (layout field removed in Astro 5)
4. **Auto-generated references**: Reference list appears automatically at the end of posts with citations

## Citation Syntax

In your markdown content, use LaTeX-style citation commands:

```markdown
This finding is supported by recent research \cite{smith2024}.
Multiple citations can be included \cite{jones2023,doe2024}.
```

## Math Syntax

Use standard LaTeX syntax for mathematical expressions:

```markdown
Inline math: $E = mc^2$

Display math:
$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$
```

## Bibliography Management

### Adding References

References are stored in `src/content/refs/library.bib` in BibTeX format:

```bibtex
@article{smith2024,
  title = {Example Research Paper},
  author = {Smith, John and Johnson, Jane},
  journal = {Journal of Examples},
  year = {2024},
  volume = {15},
  pages = {123--145},
  doi = {10.1000/example.2024.001}
}
```

### Zotero Integration (Recommended)

For automatic bibliography management:

1. **Install Zotero** with Better BibTeX plugin
2. **Configure auto-export**: Set export target to `src/content/refs/library.bib`
3. **Enable auto-sync**: Changes in Zotero automatically update your blog bibliography
4. **Consistent keys**: Use Better BibTeX for stable citation keys

### Manual Export

From Zotero:
1. Select references to export
2. Choose "Better BibTeX" format
3. Save as `src/content/refs/library.bib`
4. Commit the updated file to git

## Content Creation Workflow

### 1. Create New Post

```yaml
---
title: Your Post Title
description: Brief description
pubDate: 2024-01-01T00:00:00.000Z
author: Your Name
tags:
  - research
  - mathematics
---

# Your Post Title

Your content with math $E = mc^2$ and citations \cite{einstein1905}.
```

### 2. Write Content

- Use **Markdown headings**: `##`, `###` (not HTML)
- Use **LaTeX math**: `$...$` and `$$...$$` (not HTML spans)
- Use **clean citations**: `\cite{key}` (not manual links)

### 3. Verify Citations

- Check that all cited keys exist in `library.bib`
- Test locally with `npm run dev`
- Verify citations appear as `[1]`, `[2]`, etc.
- Confirm reference list appears at bottom

## Features

- ✅ **Automatic numbering**: Citations appear as [1], [2], etc.
- ✅ **APA 7th formatting**: References formatted according to APA guidelines  
- ✅ **Clickable links**: Citations link to reference list
- ✅ **Auto-generation**: Reference list automatically appears at post end
- ✅ **Homepage compatibility**: Citations work on individual posts and homepage excerpts
- ✅ **Bilingual support**: Works identically for English and Japanese content
- ✅ **Math rendering**: KaTeX processes LaTeX math expressions
- ✅ **Clean source**: Markdown files remain readable and maintainable

## Troubleshooting

### Citations Not Appearing?
- ✓ Check that the bibliography key exists in `library.bib`
- ✓ Verify JavaScript is enabled (citations are client-side processed)
- ✓ Test the bibliography endpoint: `curl localhost:4321/api/bibliography.json`

### Math Not Rendering?
- ✓ Use proper LaTeX syntax: `$...$` for inline, `$$...$$` for display
- ✓ Escape special characters: `\{`, `\}`, `\\`
- ✓ Check browser console for KaTeX errors

### Reference Formatting Issues?
- ✓ Check BibTeX entry completeness (author, title, year required)
- ✓ Verify special characters are properly escaped in `.bib` file
- ✓ Test individual entries by adding simple test citation

### Build Errors?
- ✓ Run `npm run build` to check for issues
- ✓ Verify no HTML artifacts remain in markdown files
- ✓ Check that frontmatter is valid YAML

## Migration from Legacy System

If you have old posts with HTML artifacts:

1. **Remove HTML headings**: `<h2>Title</h2>` → `## Title`
2. **Clean math spans**: `<span class="math">$x$</span>` → `$x$`
3. **Fix citations**: `[[1]](#ref-key)` → `\cite{key}`
4. **Remove hardcoded references**: Delete manual reference lists
5. **Remove layout field**: Not needed in Astro 5 content collections

## Technical Details

The citation system works through:

1. **Content collections**: Astro processes markdown with math and citations
2. **KaTeX rendering**: Math expressions converted to HTML during build
3. **Client-side citations**: JavaScript processes `\cite{}` on page load
4. **Bibliography API**: References served from `/api/bibliography.json`
5. **Dynamic generation**: Numbered citations and APA reference list injected

This approach ensures citations work with Astro's static generation while maintaining clean, maintainable markdown source files.