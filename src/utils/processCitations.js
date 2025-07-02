// Client-side citation processing
export function processCitationsInContent(content, bibliography) {
  // Replace \cite{key} with formatted citations
  const citeRegex = /\\cite\{([^}]+)\}/g;
  
  return content.replace(citeRegex, (match, key) => {
    if (bibliography.has(key)) {
      return `<cite data-cite="${key}"><a href="#ref-${key}" class="citation-link">[${key}]</a></cite>`;
    }
    return `<span class="citation-error">[${key}?]</span>`;
  });
}

// Initialize citation processing on page load
document.addEventListener('DOMContentLoaded', async function() {
  // Check if this page has citations
  const content = document.querySelector('.sophisticated-markdown');
  if (!content) return;
  
  const contentText = content.innerHTML;
  if (!contentText.includes('\\cite{')) return;
  
  try {
    // Load bibliography data from the server
    const response = await fetch('/api/bibliography.json');
    if (!response.ok) return;
    
    const bibliographyData = await response.json();
    const bibliography = new Map(bibliographyData);
    
    // Process citations in content
    const processedContent = processCitationsInContent(contentText, bibliography);
    content.innerHTML = processedContent;
    
    // Extract used citations
    const usedCitations = new Set();
    const citeRegex = /\\cite\{([^}]+)\}/g;
    let match;
    while ((match = citeRegex.exec(contentText)) !== null) {
      usedCitations.add(match[1]);
    }
    
    // Generate and append bibliography if citations exist
    if (usedCitations.size > 0) {
      const bibliographyHtml = generateBibliographyHtml(bibliography, usedCitations);
      content.insertAdjacentHTML('afterend', bibliographyHtml);
    }
  } catch (error) {
    console.error('Error processing citations:', error);
  }
});

function generateBibliographyHtml(bibliography, usedCitations) {
  const sortedKeys = Array.from(usedCitations).sort();
  
  let html = '<div class="bibliography"><h2 class="bibliography-title">References</h2><div class="bibliography-list">';
  
  for (const key of sortedKeys) {
    const entry = bibliography.get(key);
    if (!entry) continue;
    
    const authors = entry.entryTags.author || 'Unknown Author';
    const title = entry.entryTags.title || 'Untitled';
    const year = entry.entryTags.year || 'n.d.';
    const journal = entry.entryTags.journal || '';
    const url = entry.entryTags.url || (entry.entryTags.eprint 
      ? `https://arxiv.org/abs/${entry.entryTags.eprint}` 
      : '');

    html += `<div class="bibliography-entry" id="ref-${key}">`;
    html += `<span class="authors">${authors}</span> `;
    html += `<span class="year">(${year})</span> `;
    html += `<span class="title">${title}</span>`;
    
    if (journal) {
      html += `. <span class="journal">${journal}</span>`;
    }
    
    if (url) {
      html += `. <a href="${url}" target="_blank" rel="noopener noreferrer">Available at: ${url}</a>`;
    }
    
    html += '</div>';
  }
  
  html += '</div></div>';
  
  return html;
}