import { visit } from 'unist-util-visit';
import { loadBibliography } from './citations.js';

const CITE_PATTERN = /\\cite\{([^}]+)\}/g;

let bibliographyCache;

function cleanValue(value) {
  if (!value) return '';
  return value.replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
}

function formatAuthors(value) {
  const cleaned = cleanValue(value);
  if (!cleaned) return '';
  return cleaned.replace(/\s+and\s+/gi, ', ');
}

function splitCitations(text) {
  const parts = [];
  CITE_PATTERN.lastIndex = 0;
  let lastIndex = 0;
  let match;

  while ((match = CITE_PATTERN.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }

    const keys = match[1]
      .split(',')
      .map((key) => key.trim())
      .filter(Boolean);

    if (keys.length > 0) {
      parts.push({ type: 'cite', keys });
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return parts;
}

function hasClass(node, className) {
  if (!node || !node.properties || !node.properties.className) return false;
  const classes = node.properties.className;
  if (Array.isArray(classes)) return classes.includes(className);
  return String(classes).split(' ').includes(className);
}

function shouldSkipNode(parent) {
  if (!parent || parent.type !== 'element') return false;
  const tag = parent.tagName;
  if (tag === 'code' || tag === 'pre' || tag === 'script' || tag === 'style' || tag === 'textarea') {
    return true;
  }
  return hasClass(parent, 'katex') || hasClass(parent, 'katex-display');
}

function buildReferenceText(entryTags) {
  const authors = formatAuthors(entryTags.author);
  const year = cleanValue(entryTags.year);
  const title = cleanValue(entryTags.title);
  const container = cleanValue(entryTags.booktitle || entryTags.journal || entryTags.publisher);

  const parts = [];
  if (authors) parts.push(authors);
  if (year) parts.push(`(${year}).`);
  if (title) parts.push(`${title}.`);
  if (container) parts.push(`${container}.`);

  return parts.join(' ').trim();
}

function buildReferenceItem(key, entry) {
  const entryTags = entry?.entryTags ?? {};
  const mainText = buildReferenceText(entryTags) || `Missing reference: ${key}.`;
  const doi = cleanValue(entryTags.doi);
  const url = cleanValue(entryTags.url);

  const children = [{ type: 'text', value: mainText }];

  if (doi || url) {
    const href = doi ? `https://doi.org/${doi}` : url;
    const linkText = doi ? `doi:${doi}` : 'link';

    children.push({ type: 'text', value: ' ' });
    children.push({
      type: 'element',
      tagName: 'a',
      properties: {
        href
      },
      children: [{ type: 'text', value: linkText }]
    });
  }

  return {
    type: 'element',
    tagName: 'li',
    properties: {
      id: `ref-${key}`,
      'data-citation-key': key
    },
    children
  };
}

function buildReferencesSection(citationOrder, bibliography, sectionTitle) {
  const items = citationOrder.map((key) => buildReferenceItem(key, bibliography.get(key)));

  return {
    type: 'element',
    tagName: 'section',
    properties: {
      className: ['references']
    },
    children: [
      {
        type: 'element',
        tagName: 'h2',
        children: [{ type: 'text', value: sectionTitle }]
      },
      {
        type: 'element',
        tagName: 'ol',
        children: items
      }
    ]
  };
}

export function rehypeCitations(options = {}) {
  const { sectionTitle = 'References' } = options;

  return async (tree) => {
    if (!bibliographyCache) {
      bibliographyCache = loadBibliography();
    }

    const bibliography = await bibliographyCache;
    const citationOrder = [];
    const citationIndexByKey = new Map();

    visit(tree, 'text', (node, index, parent) => {
      if (typeof node.value !== 'string' || !parent || typeof index !== 'number') {
        return;
      }

      if (shouldSkipNode(parent)) {
        return;
      }

      const parts = splitCitations(node.value);
      const hasCitation = parts.some((part) => part.type === 'cite');
      if (!hasCitation) {
        return;
      }

      const replacementNodes = [];

      for (const part of parts) {
        if (part.type === 'text') {
          replacementNodes.push({ type: 'text', value: part.value });
          continue;
        }

        part.keys.forEach((key, keyIndex) => {
          if (!citationIndexByKey.has(key)) {
            citationIndexByKey.set(key, citationOrder.length + 1);
            citationOrder.push(key);
          }

          const citationIndex = citationIndexByKey.get(key);

          replacementNodes.push({
            type: 'element',
            tagName: 'sup',
            properties: {
              'data-citation': key,
              'data-citation-index': String(citationIndex)
            },
            children: [{ type: 'text', value: `[${citationIndex}]` }]
          });

          if (keyIndex < part.keys.length - 1) {
            replacementNodes.push({ type: 'text', value: ', ' });
          }
        });
      }

      parent.children.splice(index, 1, ...replacementNodes);
      return index + replacementNodes.length;
    });

    if (citationOrder.length > 0) {
      tree.children.push(buildReferencesSection(citationOrder, bibliography, sectionTitle));
    }
  };
}
