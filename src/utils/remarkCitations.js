import { visit } from 'unist-util-visit';
import { loadBibliography } from './citations.js';

export function remarkCitations() {
  return async function transformer(tree, file) {
    const bibliography = await loadBibliography();
    const usedCitations = new Set();
    
    // Process citation syntax \cite{key}
    visit(tree, 'text', (node) => {
      if (typeof node.value === 'string') {
        const citeRegex = /\\cite\{([^}]+)\}/g;
        let match;
        let newValue = node.value;
        
        while ((match = citeRegex.exec(node.value)) !== null) {
          const key = match[1];
          usedCitations.add(key);
          
          if (bibliography.has(key)) {
            // Replace with HTML citation
            newValue = newValue.replace(
              match[0], 
              `<cite data-cite="${key}"><a href="#ref-${key}" class="citation-link">[${key}]</a></cite>`
            );
          } else {
            newValue = newValue.replace(
              match[0], 
              `<span class="citation-error">[${key}?]</span>`
            );
          }
        }
        
        if (newValue !== node.value) {
          node.value = newValue;
          node.type = 'html';
        }
      }
    });
    
    // Store used citations in file data for later use
    if (!file.data) file.data = {};
    file.data.usedCitations = usedCitations;
    file.data.bibliography = bibliography;
  };
}