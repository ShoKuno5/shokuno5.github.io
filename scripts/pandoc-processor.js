#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class PandocProcessor {
  constructor() {
    this.projectRoot = join(__dirname, '..');
    this.bibPath = join(this.projectRoot, 'src/content/bibliography.bib');
    this.tempDir = join(this.projectRoot, '.temp');
  }

  // Process a single markdown file through Pandoc
  async processMarkdown(filePath, options = {}) {
    try {
      // Read and parse frontmatter
      const fileContent = readFileSync(filePath, 'utf-8');
      const { data: frontmatter, content } = matter(fileContent);
      
      // Check if file has citations
      const hasCitations = /[@\\]cite\{|@\w+/.test(content);
      
      if (!hasCitations) {
        console.log(`No citations found in ${filePath}, skipping Pandoc processing`);
        return fileContent; // Return original if no citations
      }

      // Convert LaTeX-style citations to Pandoc format
      let pandocContent = content;
      
      // Convert \cite{key} to [@key]
      pandocContent = pandocContent.replace(/\\cite\{([^}]+)\}/g, '[@$1]');
      
      // Prepare Pandoc command
      const pandocArgs = [
        '--from', 'markdown',
        '--to', 'html',
        '--citeproc',
        '--bibliography', this.bibPath,
        '--csl', join(this.projectRoot, 'scripts/apa.csl') // We'll create this
      ];

      // Add additional options
      if (options.citationStyle) {
        pandocArgs[pandocArgs.indexOf('--csl') + 1] = options.citationStyle;
      }

      // Create temporary input file
      const tempInput = join(this.tempDir, 'temp_input.md');
      const tempOutput = join(this.tempDir, 'temp_output.html');
      
      // Ensure temp directory exists
      execSync(`mkdir -p ${this.tempDir}`);
      
      // Write content to temp file
      writeFileSync(tempInput, pandocContent);
      
      // Run Pandoc
      const pandocCmd = `pandoc ${pandocArgs.join(' ')} "${tempInput}" -o "${tempOutput}"`;
      console.log(`Running: ${pandocCmd}`);
      
      execSync(pandocCmd);
      
      // Read processed output
      const processedHtml = readFileSync(tempOutput, 'utf-8');
      
      // Convert HTML back to markdown-like format for Astro
      const processedContent = this.htmlToAstroMarkdown(processedHtml);
      
      // Combine frontmatter with processed content
      const result = matter.stringify(processedContent, frontmatter);
      
      // Clean up temp files
      execSync(`rm -f "${tempInput}" "${tempOutput}"`);
      
      return result;
      
    } catch (error) {
      console.error(`Error processing ${filePath}:`, error.message);
      throw error;
    }
  }

  // Convert Pandoc HTML output back to Astro-compatible markdown
  htmlToAstroMarkdown(html) {
    let result = html;
    
    // Convert Pandoc citations back to a format Astro can handle
    // Pandoc generates: <a href="#ref-key" role="doc-biblioref">Author (Year)</a>
    result = result.replace(
      /<a href="#ref-([^"]+)" role="doc-biblioref">([^<]+)<\/a>/g,
      '<a href="#ref-$1" class="citation-link">[$2]</a>'
    );
    
    // Extract and format bibliography
    const bibliographyMatch = result.match(/<div id="refs" class="references[^>]*>(.*?)<\/div>/s);
    if (bibliographyMatch) {
      const bibliography = this.formatBibliography(bibliographyMatch[1]);
      result = result.replace(bibliographyMatch[0], bibliography);
    }
    
    // Clean up Pandoc's HTML structure for Astro
    result = result.replace(/<\/?div[^>]*>/g, ''); // Remove divs
    result = result.replace(/<p>/g, '').replace(/<\/p>/g, '\n\n'); // Convert p tags to markdown
    
    return result.trim();
  }

  // Format bibliography with proper styling
  formatBibliography(bibliographyHtml) {
    // Parse individual references
    const references = bibliographyHtml.match(/<div[^>]*class="csl-entry"[^>]*>.*?<\/div>/gs) || [];
    
    let formattedBib = '\n\n## References\n\n';
    formattedBib += '<div class="bibliography">\n';
    formattedBib += '<div class="bibliography-list">\n';
    
    references.forEach((ref, index) => {
      const number = index + 1;
      const cleanRef = ref.replace(/<\/?div[^>]*>/g, '').trim();
      
      formattedBib += `<div class="bibliography-entry">\n`;
      formattedBib += `  <div class="citation-layout">\n`;
      formattedBib += `    <span class="citation-number">[${number}]</span>\n`;
      formattedBib += `    <div class="citation-content">${cleanRef}</div>\n`;
      formattedBib += `  </div>\n`;
      formattedBib += `</div>\n`;
    });
    
    formattedBib += '</div>\n</div>\n';
    
    return formattedBib;
  }

  // Download APA CSL style if not exists
  async ensureCSLStyle() {
    const cslPath = join(this.projectRoot, 'scripts/apa.csl');
    
    if (!existsSync(cslPath)) {
      console.log('Downloading APA CSL style...');
      try {
        execSync(`curl -o "${cslPath}" "https://raw.githubusercontent.com/citation-style-language/styles/master/apa.csl"`);
        console.log('APA CSL style downloaded successfully');
      } catch (error) {
        console.error('Failed to download APA CSL style:', error.message);
        throw error;
      }
    }
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node pandoc-processor.js <markdown-file>');
    process.exit(1);
  }
  
  const processor = new PandocProcessor();
  
  try {
    // Ensure CSL style is available
    await processor.ensureCSLStyle();
    
    const inputFile = args[0];
    const outputFile = args[1] || inputFile;
    
    console.log(`Processing ${inputFile} with Pandoc...`);
    const result = await processor.processMarkdown(inputFile);
    
    writeFileSync(outputFile, result);
    console.log(`Processed file saved to ${outputFile}`);
    
  } catch (error) {
    console.error('Processing failed:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export default PandocProcessor;