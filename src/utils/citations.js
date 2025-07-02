import bibtexParse from 'bibtex-parse-js';
import fs from 'fs';
import path from 'path';

// Load bibliography from .bib file
export async function loadBibliography() {
  try {
    const bibPath = path.join(process.cwd(), 'src/content/bibliography.bib');
    const bibContent = fs.readFileSync(bibPath, 'utf-8');
    
    // Parse BibTeX
    const parsed = bibtexParse.toJSON(bibContent);
    const bibliography = new Map();
    
    parsed.forEach(entry => {
      bibliography.set(entry.citationKey, entry);
    });
    
    return bibliography;
  } catch (error) {
    console.error('Error loading bibliography:', error);
    return new Map();
  }
}

// Format citation for inline use (APA style with numbers)
export function formatCitation(key, citationNumber) {
  return `<cite data-cite="${key}"><a href="#ref-${key}" class="citation-link">[${citationNumber}]</a></cite>`;
}

// Generate bibliography entry HTML in APA 7th edition format
export function formatBibliographyEntry(entry) {
  const authors = formatAuthorsAPA(entry.entryTags.author || 'Unknown Author');
  const title = cleanTitle(entry.entryTags.title || 'Untitled');
  const year = entry.entryTags.year || 'n.d.';
  const entryType = entry.entryType || 'article';
  
  // Handle different entry types
  const journal = entry.entryTags.journal || '';
  const booktitle = entry.entryTags.booktitle || '';
  const publisher = entry.entryTags.publisher || '';
  const volume = entry.entryTags.volume || '';
  const number = entry.entryTags.number || '';
  const pages = entry.entryTags.pages || '';
  const doi = entry.entryTags.doi || '';
  const note = entry.entryTags.note || '';
  
  // Handle URLs and arXiv
  let url = entry.entryTags.url || '';
  if (doi && doi.includes('arXiv')) {
    const arxivId = doi.split('arXiv.')[1];
    url = `https://arxiv.org/abs/${arxivId}`;
  }

  let formatted = `<div class="bibliography-entry" id="ref-${entry.citationKey}">`;
  
  // Author (Year). Title. 
  formatted += `<span class="authors">${authors}</span> `;
  formatted += `<span class="year">(${year})</span>. `;
  
  // Handle different publication types
  if (entryType === 'article' && journal) {
    // Journal article
    formatted += `<span class="title">${title}</span>. `;
    formatted += `<span class="journal"><em>${journal}</em></span>`;
    
    if (volume) {
      formatted += `, <span class="volume"><em>${volume}</em></span>`;
      if (number) {
        formatted += `<span class="issue">(${number})</span>`;
      }
    }
    
    if (pages) {
      formatted += `, <span class="pages">${formatPagesAPA(pages)}</span>`;
    }
  } else if (entryType === 'inproceedings' && booktitle) {
    // Conference proceedings
    formatted += `<span class="title">${title}</span>. `;
    formatted += `In <span class="booktitle"><em>${booktitle}</em></span>`;
    
    if (volume) {
      formatted += ` (Vol. ${volume})`;
    }
    
    if (pages) {
      formatted += ` (pp. ${formatPagesAPA(pages)})`;
    }
    
    if (publisher) {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  } else if (entryType === 'misc' || publisher === 'arXiv') {
    // ArXiv preprints and misc entries
    formatted += `<span class="title">${title}</span>`;
    
    if (publisher === 'arXiv') {
      formatted += `. <em>arXiv preprint</em>`;
    } else if (publisher && publisher !== 'arXiv') {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  } else {
    // Default formatting for other types
    formatted += `<span class="title"><em>${title}</em></span>`;
    
    if (publisher) {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  }
  
  // Add DOI or URL
  if (doi) {
    if (doi.includes('arXiv')) {
      formatted += `. <a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-url">${url}</a>`;
    } else {
      formatted += `. <a href="https://doi.org/${doi}" target="_blank" rel="noopener noreferrer" class="citation-url">https://doi.org/${doi}</a>`;
    }
  } else if (url) {
    formatted += `. <a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-url">${url}</a>`;
  }
  
  formatted += `</div>`;
  
  return formatted;
}

// Format authors according to APA 7th edition
function formatAuthorsAPA(authorString) {
  if (!authorString) return 'Unknown Author';
  
  // Split multiple authors (assuming "and" or "&" separation)
  const authors = authorString.split(/ and | & |, and |, & /).map(author => author.trim());
  
  if (authors.length === 1) {
    return formatSingleAuthorAPA(authors[0]);
  } else if (authors.length === 2) {
    return `${formatSingleAuthorAPA(authors[0])}, & ${formatSingleAuthorAPA(authors[1])}`;
  } else if (authors.length <= 20) {
    const formattedAuthors = authors.slice(0, -1).map(formatSingleAuthorAPA);
    return `${formattedAuthors.join(', ')}, & ${formatSingleAuthorAPA(authors[authors.length - 1])}`;
  } else {
    // For 21+ authors, list first 19, then "...", then last author
    const formattedAuthors = authors.slice(0, 19).map(formatSingleAuthorAPA);
    return `${formattedAuthors.join(', ')}, ... ${formatSingleAuthorAPA(authors[authors.length - 1])}`;
  }
}

// Format single author: Last, F. M.
function formatSingleAuthorAPA(author) {
  // Handle "Last, First Middle" or "First Middle Last" formats
  if (author.includes(',')) {
    const [last, first] = author.split(',').map(s => s.trim());
    return `${last}, ${getInitials(first)}`;
  } else {
    const parts = author.split(' ').filter(part => part.length > 0);
    if (parts.length === 1) return parts[0];
    const last = parts[parts.length - 1];
    const firstMiddle = parts.slice(0, -1).join(' ');
    return `${last}, ${getInitials(firstMiddle)}`;
  }
}

// Get initials: "First Middle" -> "F. M."
function getInitials(names) {
  return names.split(' ')
    .filter(name => name.length > 0)
    .map(name => name.charAt(0).toUpperCase() + '.')
    .join(' ');
}

// Format page ranges according to APA style
function formatPagesAPA(pages) {
  if (pages.includes('-') || pages.includes('–')) {
    // Page range
    const [start, end] = pages.split(/[-–]/).map(p => p.trim());
    return `${start}–${end}`;
  }
  return pages;
}

// Clean title formatting (remove double braces, etc.)
function cleanTitle(title) {
  if (!title) return 'Untitled';
  
  // Remove double braces {{}} and convert to proper case
  let cleaned = title.replace(/\{\{([^}]+)\}\}/g, '$1');
  
  // Remove any remaining single braces
  cleaned = cleaned.replace(/[{}]/g, '');
  
  // Clean up any extra spaces
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  
  return cleaned;
}

// Process citations in markdown content
export function processCitations(content, bibliography) {
  // First pass: extract all citation keys to assign numbers
  const usedCitations = extractCitationKeys(content);
  const sortedKeys = Array.from(usedCitations).sort();
  const citationNumbers = new Map();
  
  // Assign numbers to citations in order of appearance
  sortedKeys.forEach((key, index) => {
    citationNumbers.set(key, index + 1);
  });
  
  // Replace \cite{key} with formatted citations
  const citeRegex = /\\cite\{([^}]+)\}/g;
  
  return content.replace(citeRegex, (match, key) => {
    if (bibliography.has(key)) {
      const citationNumber = citationNumbers.get(key);
      return formatCitation(key, citationNumber);
    }
    return `<span class="citation-error">[${key}?]</span>`;
  });
}

// Generate bibliography section with numbers
export function generateBibliography(bibliography, usedCitations) {
  let bibHtml = '<div class="bibliography"><h2 class="bibliography-title">References</h2><div class="bibliography-list">';
  
  // Sort citations alphabetically by key
  const sortedKeys = Array.from(usedCitations).sort();
  
  sortedKeys.forEach((key, index) => {
    if (bibliography.has(key)) {
      const entry = bibliography.get(key);
      const citationNumber = index + 1;
      bibHtml += formatBibliographyEntryWithNumber(entry, citationNumber);
    }
  });
  
  bibHtml += '</div></div>';
  
  return bibHtml;
}

// Generate bibliography entry with number prefix
function formatBibliographyEntryWithNumber(entry, number) {
  const authors = formatAuthorsAPA(entry.entryTags.author || 'Unknown Author');
  const title = cleanTitle(entry.entryTags.title || 'Untitled');
  const year = entry.entryTags.year || 'n.d.';
  const entryType = entry.entryType || 'article';
  
  // Handle different entry types
  const journal = entry.entryTags.journal || '';
  const booktitle = entry.entryTags.booktitle || '';
  const publisher = entry.entryTags.publisher || '';
  const volume = entry.entryTags.volume || '';
  const entryNumber = entry.entryTags.number || '';
  const pages = entry.entryTags.pages || '';
  const doi = entry.entryTags.doi || '';
  const note = entry.entryTags.note || '';
  
  // Handle URLs and arXiv
  let url = entry.entryTags.url || '';
  if (doi && doi.includes('arXiv')) {
    const arxivId = doi.split('arXiv.')[1];
    url = `https://arxiv.org/abs/${arxivId}`;
  }

  let formatted = `<div class="bibliography-entry" id="ref-${entry.citationKey}">`;
  formatted += `<span class="citation-number">[${number}]</span>`;
  formatted += `<span class="citation-content">`;
  
  // Author (Year). Title. 
  formatted += `<span class="authors">${authors}</span> `;
  formatted += `<span class="year">(${year})</span>. `;
  
  // Handle different publication types
  if (entryType === 'article' && journal) {
    // Journal article
    formatted += `<span class="title">${title}</span>. `;
    formatted += `<span class="journal"><em>${journal}</em></span>`;
    
    if (volume) {
      formatted += `, <span class="volume"><em>${volume}</em></span>`;
      if (entryNumber) {
        formatted += `<span class="issue">(${entryNumber})</span>`;
      }
    }
    
    if (pages) {
      formatted += `, <span class="pages">${formatPagesAPA(pages)}</span>`;
    }
  } else if (entryType === 'inproceedings' && booktitle) {
    // Conference proceedings
    formatted += `<span class="title">${title}</span>. `;
    formatted += `In <span class="booktitle"><em>${booktitle}</em></span>`;
    
    if (volume) {
      formatted += ` (Vol. ${volume})`;
    }
    
    if (pages) {
      formatted += ` (pp. ${formatPagesAPA(pages)})`;
    }
    
    if (publisher) {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  } else if (entryType === 'misc' || publisher === 'arXiv') {
    // ArXiv preprints and misc entries
    formatted += `<span class="title">${title}</span>`;
    
    if (publisher === 'arXiv') {
      formatted += `. <em>arXiv preprint</em>`;
    } else if (publisher && publisher !== 'arXiv') {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  } else {
    // Default formatting for other types
    formatted += `<span class="title"><em>${title}</em></span>`;
    
    if (publisher) {
      formatted += `. <span class="publisher">${publisher}</span>`;
    }
  }
  
  // Add DOI or URL
  if (doi) {
    if (doi.includes('arXiv')) {
      formatted += `. <a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-url">${url}</a>`;
    } else {
      formatted += `. <a href="https://doi.org/${doi}" target="_blank" rel="noopener noreferrer" class="citation-url">https://doi.org/${doi}</a>`;
    }
  } else if (url) {
    formatted += `. <a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-url">${url}</a>`;
  }
  
  formatted += `</span></div>`;
  
  return formatted;
}

// Extract citation keys from content
export function extractCitationKeys(content) {
  const citeRegex = /\\cite\{([^}]+)\}/g;
  const keys = new Set();
  let match;
  
  while ((match = citeRegex.exec(content)) !== null) {
    keys.add(match[1]);
  }
  
  return keys;
}