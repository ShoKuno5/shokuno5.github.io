#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

/**
 * Generates a table of contents for a Markdown file
 * @param {string} filePath - Path to the markdown file
 */
function generateTOC(filePath) {
  try {
    const content = readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    
    // Find frontmatter boundaries
    let frontmatterEnd = -1;
    let inFrontmatter = false;
    let frontmatterStart = -1;
    
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim() === '---') {
        if (!inFrontmatter && frontmatterStart === -1) {
          frontmatterStart = i;
          inFrontmatter = true;
        } else if (inFrontmatter) {
          frontmatterEnd = i;
          break;
        }
      }
    }
    
    // Extract headers (excluding frontmatter and TOC section)
    const headers = [];
    const startLine = frontmatterEnd > -1 ? frontmatterEnd + 1 : 0;
    
    // Find where the actual content starts (after TOC section)
    let contentStartLine = startLine;
    let foundTocEnd = false;
    
    for (let i = startLine; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Look for TOC header
      if (line.match(/^#{1,6}\s+.*table of contents/i)) {
        // Find the end of TOC section (marked by ---)
        for (let j = i + 1; j < lines.length; j++) {
          if (lines[j].trim() === '---') {
            contentStartLine = j + 1;
            foundTocEnd = true;
            break;
          }
        }
        break;
      }
    }
    
    for (let i = contentStartLine; i < lines.length; i++) {
      const line = lines[i].trim();
      const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
      
      if (headerMatch) {
        const level = headerMatch[1].length;
        const text = headerMatch[2];

        // Skip any heading that itself is the TOC title
        const plainText = text.replace(/[*_`~]/g, '').trim();
        if (/^table of contents$/i.test(plainText)) {
          continue;
        }
        
        const anchor = text.toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .replace(/^-|-$/g, '');
        
        headers.push({ level, text, anchor });
      }
    }
    
    if (headers.length === 0) {
      console.log('No headers found in the file.');
      return;
    }
    
    // Generate TOC
    let toc = '## Table of Contents\n\n';
    
    headers.forEach(header => {
      const indent = '  '.repeat(header.level - 1);
      toc += `${indent}- [${header.text}](#${header.anchor})\n`;
    });
    
    toc += '\n---\n';
    
    // Check if TOC already exists (any heading level)
    // Matches from a line like `# Table of Contents` (any 1-6 #) up to the next `---` line
    const tocRegex = /^#{1,6}\s+table of contents[\s\S]*?^---\s*$\n?/gim;
    let newContent;
    
    if (tocRegex.test(content)) {
      // Replace the first existing TOC and remove any duplicates
      let replaced = false;
      newContent = content.replace(tocRegex, () => {
        if (!replaced) {
          replaced = true;
          return toc;
        }
        return '';
      });
      console.log('Updated existing table of contents (and removed duplicates).');
    } else {
      // Insert TOC after frontmatter
      const beforeToc = lines.slice(0, frontmatterEnd + 1).join('\n');
      const afterToc = lines.slice(frontmatterEnd + 1).join('\n');
      newContent = beforeToc + '\n\n' + toc + '\n' + afterToc;
      console.log('Added new table of contents.');
    }
    
    writeFileSync(filePath, newContent);
    console.log(`TOC generated successfully for ${filePath}`);
    
  } catch (error) {
    console.error('Error generating TOC:', error.message);
  }
}

// Get file path from command line arguments
const filePath = process.argv[2];

if (!filePath) {
  console.error('Usage: node generate-toc.js <markdown-file-path>');
  process.exit(1);
}

const resolvedPath = resolve(filePath);
generateTOC(resolvedPath);
