import fs from 'node:fs';
import { visit } from 'unist-util-visit';
import { loadBibliography } from './citations.js';

const CITE_PATTERN = /\\cite\{([^}]+)\}/g;

let bibliographyCache;

function hashToken(value) {
  let hash = 0;
  const input = String(value || '');
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
}

function slugToken(value) {
  const slug = String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'item';
}

function createDocNamespace(file) {
  const pathFromFile = typeof file?.path === 'string' ? file.path : '';
  if (!pathFromFile) return 'doc';
  const normalized = pathFromFile.replace(/\\/g, '/');
  const baseName = normalized.split('/').pop() || normalized;
  const withoutExt = baseName.replace(/\.[^.]+$/, '');
  return `${slugToken(withoutExt)}-${hashToken(normalized)}`;
}

function createCitationToken(key) {
  return `${slugToken(key)}-${hashToken(key)}`;
}

function getReferenceId(docNamespace, key) {
  return `ref-${docNamespace}-${createCitationToken(key)}`;
}

function getCitationId(docNamespace, key, occurrenceIndex) {
  return `cite-${docNamespace}-${createCitationToken(key)}-${occurrenceIndex}`;
}

function getFileText(file) {
  if (!file) return '';
  if (typeof file.value === 'string') return file.value;
  if (file.value != null) return String(file.value);
  const filePath = typeof file.path === 'string' ? file.path : '';
  if (!filePath) return '';
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch {
    return '';
  }
}

function buildLineOffsets(text) {
  const offsets = [0];
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === '\n') offsets.push(i + 1);
  }
  return offsets;
}

function lineNumberFromOffset(offsets, offset) {
  if (!offsets || typeof offset !== 'number') return null;
  let low = 0;
  let high = offsets.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const current = offsets[mid];
    const next = offsets[mid + 1];
    if (current <= offset && (next === undefined || next > offset)) {
      return mid + 1;
    }
    if (current > offset) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return offsets.length;
}

function cleanValue(value) {
  if (!value) return '';
  return value.replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
}

function formatAuthors(value) {
  const cleaned = cleanValue(value);
  if (!cleaned) return '';
  return cleaned.replace(/\s+and\s+/gi, ', ');
}

function buildCitationPreview(text, matchIndex, matchLength) {
  const windowSize = 60;
  const start = Math.max(0, matchIndex - windowSize);
  const end = Math.min(text.length, matchIndex + matchLength + windowSize);
  const snippet = text.slice(start, end);
  const cleaned = snippet.replace(CITE_PATTERN, '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  const prefix = start > 0 ? '…' : '';
  const suffix = end < text.length ? '…' : '';
  const combined = `${prefix}${cleaned}${suffix}`;
  return combined.length > 140 ? `${combined.slice(0, 137)}…` : combined;
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
      parts.push({
        type: 'cite',
        keys,
        matchIndex: match.index,
        matchLength: match[0].length
      });
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

function buildReferenceItem(docNamespace, key, entry) {
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
      id: getReferenceId(docNamespace, key),
      'data-citation-key': key
    },
    children
  };
}

function buildReferenceBackrefs(occurrences) {
  if (!occurrences || occurrences.length === 0) return null;
  const links = occurrences.map((occurrence, index) => {
    const label = occurrence.label || String(index + 1);
    const ariaLabel = `Back to line ${label}`;
    const previewText = occurrence.preview || '';
    const properties = {
      href: `#${occurrence.id}`,
      className: ['citation-backref'],
      'aria-label': ariaLabel,
      'data-line': label
    };
    if (previewText) {
      properties['data-preview'] = previewText;
    }
    return {
      type: 'element',
      tagName: 'a',
      properties,
      children: [{ type: 'text', value: `↩︎${label}` }]
    };
  });

  return {
    type: 'element',
    tagName: 'span',
    properties: {
      className: ['citation-backrefs']
    },
    children: links
  };
}

function buildReferencesSection(
  docNamespace,
  citationOrder,
  bibliography,
  sectionTitle,
  citationOccurrencesByKey
) {
  const items = citationOrder.map((key) => {
    const item = buildReferenceItem(docNamespace, key, bibliography.get(key));
    const backrefs = buildReferenceBackrefs(citationOccurrencesByKey.get(key));
    if (backrefs) {
      item.children.push({ type: 'text', value: ' ' }, backrefs);
    }
    return item;
  });

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

  return async (tree, file) => {
    if (!bibliographyCache) {
      bibliographyCache = loadBibliography();
    }

    const bibliography = await bibliographyCache;
    const citationOrder = [];
    const citationIndexByKey = new Map();
    const citationOccurrencesByKey = new Map();
    const docNamespace = createDocNamespace(file);
    const sourceText = getFileText(file);
    const lineOffsets = sourceText ? buildLineOffsets(sourceText) : null;
    let sourceSearchIndex = 0;

    const resolveLineNumber = (node, part) => {
      const position = node.position?.start;
      if (typeof position?.line === 'number') {
        return position.line;
      }

      if (lineOffsets && typeof position?.offset === 'number') {
        const offset = position.offset + (part.matchIndex ?? 0);
        const matchLength = part.matchLength ?? 0;
        const nextIndex = offset + matchLength;
        if (nextIndex > sourceSearchIndex) {
          sourceSearchIndex = nextIndex;
        }
        return lineNumberFromOffset(lineOffsets, offset);
      }

      if (lineOffsets && sourceText && typeof node.value === 'string') {
        const start = part.matchIndex ?? 0;
        const length = part.matchLength ?? 0;
        const token = node.value.slice(start, start + length);
        if (token) {
          const found = sourceText.indexOf(token, sourceSearchIndex);
          if (found !== -1) {
            sourceSearchIndex = found + token.length;
            return lineNumberFromOffset(lineOffsets, found);
          }
        }
      }

      return null;
    };

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

        const lineNumber = resolveLineNumber(node, part);
        const preview = buildCitationPreview(
          node.value,
          part.matchIndex ?? 0,
          part.matchLength ?? 0
        );

        part.keys.forEach((key, keyIndex) => {
          if (!citationIndexByKey.has(key)) {
            citationIndexByKey.set(key, citationOrder.length + 1);
            citationOrder.push(key);
          }

          const citationIndex = citationIndexByKey.get(key);
          const occurrences = citationOccurrencesByKey.get(key) || [];
          const occurrenceIndex = occurrences.length + 1;
          const citeId = getCitationId(docNamespace, key, occurrenceIndex);
          const label = lineNumber ? String(lineNumber) : String(occurrenceIndex);
          occurrences.push({
            id: citeId,
            label,
            preview
          });
          citationOccurrencesByKey.set(key, occurrences);

          replacementNodes.push({
            type: 'element',
            tagName: 'a',
            properties: {
              href: `#${getReferenceId(docNamespace, key)}`,
              className: ['citation-link'],
              'aria-label': `Reference ${citationIndex}`,
              'data-citation': key,
              'data-citation-index': String(citationIndex),
              'data-citation-line': label,
              id: citeId
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
      tree.children.push(
        buildReferencesSection(
          docNamespace,
          citationOrder,
          bibliography,
          sectionTitle,
          citationOccurrencesByKey
        )
      );
    }
  };
}
