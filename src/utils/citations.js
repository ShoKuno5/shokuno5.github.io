import fs from 'fs/promises';
import path from 'path';

// Simple BibTeX parser for essential entries
function parseBibTeX(content) {
  const bibliography = new Map();
  
  // Match @type{key, entries}
  const entryRegex = /@(\w+)\{([^,]+),\s*([\s\S]*?)\n\}/g;
  
  let match;
  while ((match = entryRegex.exec(content)) !== null) {
    const [, entryType, key, content] = match;
    
    const entryTags = {};
    // Parse key-value pairs with proper brace handling
    const tagRegex = /(\w+)\s*=\s*\{/g;
    let tagMatch;
    while ((tagMatch = tagRegex.exec(content)) !== null) {
      const fieldName = tagMatch[1];
      const startPos = tagMatch.index + tagMatch[0].length;
      
      // Find the matching closing brace
      let braceCount = 1;
      let endPos = startPos;
      while (endPos < content.length && braceCount > 0) {
        if (content[endPos] === '{') braceCount++;
        else if (content[endPos] === '}') braceCount--;
        endPos++;
      }
      
      if (braceCount === 0) {
        const fieldValue = content.slice(startPos, endPos - 1);
        entryTags[fieldName] = fieldValue;
      }
    }
    
    bibliography.set(key, {
      entryType,
      entryTags
    });
  }
  
  return bibliography;
}

export async function loadBibliography() {
  try {
    const bibPath = path.join(process.cwd(), 'src/content/bibliography.bib');
    const content = await fs.readFile(bibPath, 'utf-8');
    return parseBibTeX(content);
  } catch (error) {
    console.error('Error loading bibliography:', error);
    return new Map();
  }
}