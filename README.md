# Myfolio: A Performance-Optimized Bilingual Technical Blog

A high-performance, statically-generated technical blog built with Astro, featuring complete bilingual support, academic citation management, and an innovative content-first architecture.

## 🎯 Design Philosophy

### 1. **Content-First Architecture**
All content lives in markdown collections, separating concerns between content management and presentation logic. This approach enables:
- Version-controlled content management
- Easy content updates without touching component code
- Consistent bilingual content structure
- Clear separation of data and presentation layers

### 2. **Performance as a Feature**
Every architectural decision prioritizes performance:
- Static generation with zero client-side JavaScript by default
- Lazy loading for images with custom rehype plugin
- CSS inlining to eliminate render-blocking resources
- Asset optimization with content hashing
- Aggressive minification and dead code elimination

### 3. **True Bilingual Support**
Not just translated UI elements, but complete content parity:
- Separate content directories for each language
- URL-based language routing without redirects
- Language-aware navigation components
- Consistent content structure across languages

## 🛠 Technical Stack

### Core Framework
- **Astro 5.10** - Static site generator with island architecture
- **TypeScript** - Type safety for components and utilities
- **Tailwind CSS** - Utility-first styling with JIT compilation

### Content Processing
- **MDX** - Enhanced markdown with component support
- **KaTeX** - Fast math rendering for technical content
- **remark/rehype** - Extensible markdown processing pipeline
- **Citation.js** - Academic citation management with BibTeX support

### Build Optimization
- **Terser** - Advanced JavaScript minification
- **PostCSS** - CSS optimization and processing
- **Vite** - Next-generation bundling with tree-shaking

## 🏗 Architecture Deep Dive

### Content Collections

```typescript
// Content collection structure
src/content/
├── config.ts           // Collection schemas
├── posts/              // Blog posts
│   ├── en/            // English posts
│   └── ja/            // Japanese posts
├── about/             // About page content
├── projects/          // Project details
├── projects-page/     // Projects index content
├── research/          // Research page content
├── persona/           // Persona page content
└── naive-hope/        // Protected content
```

Each collection uses Zod schemas for type-safe frontmatter validation:

```typescript
// Example schema from config.ts
const postsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    tags: z.array(z.string()),
    lang: z.enum(['en', 'ja']),
    // ... additional fields
  })
});
```

### Bilingual Implementation

The i18n system uses Astro's built-in internationalization with custom enhancements:

```javascript
// astro.config.mjs
i18n: {
  defaultLocale: 'en',
  locales: ['en', 'ja'],
  routing: {
    prefixDefaultLocale: false  // Clean URLs for default language
  }
}
```

Language detection and routing logic:
- URL parsing for `/ja/` prefix detection
- Dynamic navigation component switching
- Content collection filtering by language

### Performance Optimizations

#### 1. **Lazy Image Loading**
Custom rehype plugin for automatic lazy loading:
```javascript
// src/utils/rehype-lazy-images.js
export function rehypeLazyImages() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img') {
        node.properties.loading = 'lazy';
        node.properties.decoding = 'async';
      }
    });
  };
}
```

#### 2. **CSS Strategy**
- Inline critical CSS to eliminate render-blocking
- Tailwind JIT mode for minimal CSS footprint
- Single CSS bundle to reduce HTTP requests

#### 3. **Asset Optimization**
```javascript
// Vite configuration for optimal chunking
rollupOptions: {
  output: {
    manualChunks: undefined,  // Prevent over-chunking
    assetFileNames: 'assets/[name].[hash][extname]',
    // Content-based hashing for cache optimization
  }
}
```

### Citation System

Academic citation support with BibTeX:
- Server-side bibliography parsing
- Client-side citation rendering (when needed)
- Support for multiple citation styles
- Markdown-friendly citation syntax

### Build Pipeline

```javascript
// package.json scripts
{
  "prebuild": "node scripts/prebuild.js",  // Pre-processing
  "build": "npm run prebuild && astro check && astro build"
}
```

The prebuild script handles:
- Content validation
- Asset optimization
- Environment setup

## 🔄 Development Workflow

### Local Development
```bash
npm run dev         # Start dev server with HMR
npm run build       # Production build
npm run preview     # Preview production build
```

### Content Management
```bash
# Translation workflow
node translate.js <filename>  # Translate English post to Japanese
```

### Git Workflow
- Feature branches for new content/features
- Automated commit messages with AI assistance
- Performance commits tracked separately

## 📊 Performance Metrics

Target metrics achieved:
- **Lighthouse Score**: 95+ across all categories
- **First Contentful Paint**: < 1s
- **Time to Interactive**: < 1.5s
- **Bundle Size**: < 200KB initial load

## 🚀 Future Enhancements

### Planned Features
1. **Enhanced Search**: Full-text search with language awareness
2. **RSS Feeds**: Separate feeds per language
3. **Reading Time**: Accurate estimates for bilingual content
4. **Dark Mode**: System-aware theme switching
5. **Analytics**: Privacy-focused usage tracking

### Research Automation
Experimental AI-powered research system for:
- Content improvement suggestions
- Citation discovery
- Performance optimization opportunities
- Automated testing scenarios

## 🤝 Contributing

This is a personal portfolio project, but architectural patterns and performance techniques are open for discussion and learning.

### Key Principles for Contributors
1. Maintain content-first architecture
2. Ensure complete bilingual parity
3. Prioritize performance in all changes
4. Follow existing code patterns
5. Write meaningful commit messages

## 📝 License

This project showcases technical implementation patterns. Content rights reserved, code patterns available for learning purposes.

## 🤖 Agentic Development

This site represents a modern approach to web development, created with the aid of AI-powered agentic coding. Key aspects include:

### AI-Assisted Development
- **Claude AI Integration**: Leveraging Claude for code generation, optimization, and architectural decisions
- **Automated Code Review**: AI-driven analysis for performance and best practices
- **Intelligent Refactoring**: Pattern recognition and code improvement suggestions
- **Content Generation**: AI-assisted translation and content enhancement

### Benefits of Agentic Approach
1. **Accelerated Development**: Complex features implemented with AI collaboration
2. **Consistency**: AI ensures consistent code patterns across the codebase
3. **Best Practices**: Automatic adherence to modern web standards
4. **Performance Optimization**: AI-suggested optimizations based on real metrics
5. **Documentation**: Comprehensive docs generated alongside code

This project demonstrates the future of software development where human creativity and AI capabilities combine to create high-quality, performant applications.

---

**Built with focus on performance, accessibility, and developer experience through human-AI collaboration.**