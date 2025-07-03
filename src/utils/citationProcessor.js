// Server-side citation processing utilities
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load bibliography data
let bibliographyData = null;
async function loadBibliography() {
  if (!bibliographyData) {
    try {
      const bibPath = path.join(__dirname, '../content/bibliography.bib');
      const bibContent = fs.readFileSync(bibPath, 'utf-8');
      bibliographyData = parseBibliography(bibContent);
    } catch (error) {
      console.error('Failed to load bibliography:', error);
      bibliographyData = new Map();
    }
  }
  return bibliographyData;
}

// Parse bibliography from BibTeX content
function parseBibliography(bibContent) {
  const bibliography = new Map();
  const entryRegex = /@\w+\{([^,]+),\s*([^}]+)\}/gs;
  let match;
  
  while ((match = entryRegex.exec(bibContent)) !== null) {
    const key = match[1].trim();
    const fields = match[2];
    
    const entry = { key };
    
    // Parse individual fields
    const fieldRegex = /(\w+)\s*=\s*[{"]([^}}"]+)[}"]/g;
    let fieldMatch;
    
    while ((fieldMatch = fieldRegex.exec(fields)) !== null) {
      const fieldName = fieldMatch[1].toLowerCase();
      const fieldValue = fieldMatch[2];
      entry[fieldName] = fieldValue;
    }
    
    bibliography.set(key, entry);
  }
  
  return bibliography;
}

// Process citations in HTML content
export async function processCitations(htmlContent) {
  const bibliography = await loadBibliography();
  
  // Extract all citation keys
  const usedCitations = new Set();
  const citeRegex = /\\cite\{([^}]+)\}/g;
  let match;
  
  while ((match = citeRegex.exec(htmlContent)) !== null) {
    const key = match[1];
    if (bibliography.has(key)) {
      usedCitations.add(key);
    }
  }
  
  if (usedCitations.size === 0) {
    return htmlContent;
  }
  
  // Sort citations and assign numbers
  const sortedKeys = Array.from(usedCitations).sort();
  const citationNumbers = new Map();
  sortedKeys.forEach((key, index) => {
    citationNumbers.set(key, index + 1);
  });
  
  // Replace citations with numbered links
  let processedContent = htmlContent;
  for (const [key, number] of citationNumbers) {
    const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`\\\\cite\\{${escapedKey}\\}`, 'g');
    const replacement = `<cite data-cite="${key}"><a href="#ref-${key}" class="citation-link">[${number}]</a></cite>`;
    processedContent = processedContent.replace(regex, replacement);
  }
  
  // Generate bibliography HTML
  const bibliographyHtml = generateBibliographyHtml(bibliography, usedCitations);
  processedContent += bibliographyHtml;
  
  return processedContent;
}

// Generate bibliography HTML
function generateBibliographyHtml(bibliography, usedCitations) {
  if (usedCitations.size === 0) {
    return '';
  }
  
  const sortedKeys = Array.from(usedCitations).sort();
  let html = `
    <div class="bibliography">
      <h2>References</h2>
      <ol class="bibliography-list">
  `;
  
  sortedKeys.forEach((key, index) => {
    const entry = bibliography.get(key);
    if (entry) {
      const citationNumber = index + 1;
      const formattedEntry = formatBibliographyEntry(entry);
      html += `
        <li id="ref-${key}" class="bibliography-item">
          <span class="citation-number">[${citationNumber}]</span>
          <span class="citation-content">${formattedEntry}</span>
        </li>
      `;
    }
  });
  
  html += `
      </ol>
    </div>
  `;
  
  return html;
}

// Format bibliography entry in APA style
function formatBibliographyEntry(entry) {
  let formatted = '';
  
  if (entry.author) {
    formatted += entry.author;
  }
  
  if (entry.year) {
    formatted += ` (${entry.year})`;
  }
  
  if (entry.title) {
    formatted += `. <em>${entry.title}</em>`;
  }
  
  if (entry.journal) {
    formatted += `. ${entry.journal}`;
  } else if (entry.booktitle) {
    formatted += `. In ${entry.booktitle}`;
  }
  
  if (entry.volume && entry.pages) {
    formatted += `, ${entry.volume}, ${formatPages(entry.pages)}`;
  } else if (entry.pages) {
    formatted += `, ${formatPages(entry.pages)}`;
  }
  
  if (entry.publisher) {
    formatted += `. ${entry.publisher}`;
  }
  
  if (entry.url) {
    formatted += `. <a href="${entry.url}" target="_blank" rel="noopener noreferrer">${entry.url}</a>`;
  }
  
  return formatted + '.';
}

// Format page numbers
function formatPages(pages) {
  if (pages.includes('-') || pages.includes('–')) {
    const parts = pages.split(/[-–]/).map(p => p.trim());
    return `pp. ${parts[0]}–${parts[1]}`;
  }
  return `p. ${pages}`;
}
