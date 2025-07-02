#!/usr/bin/env node

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';
import { execSync } from 'child_process';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
  const projectRoot = join(__dirname, '..');
  const contentDir = join(projectRoot, 'src/content/posts');
  const bibPath = join(projectRoot, 'src/content/bibliography.bib');
  const cslPath = join(projectRoot, 'scripts/apa.csl');
  
  console.log('🔬 Pandoc Citation Preprocessing');
  console.log('================================');
  
  // Ensure APA CSL style exists
  await ensureCSLStyle(cslPath);
  
  // Process all markdown files
  await processDirectory(contentDir, bibPath, cslPath);
  
  console.log('✅ Citation preprocessing completed!');
}

async function ensureCSLStyle(cslPath) {
  try {
    readFileSync(cslPath);
    console.log('✅ APA CSL style found');
  } catch {
    console.log('📥 Downloading APA CSL style...');
    try {
      execSync(`mkdir -p "$(dirname "${cslPath}")"`);
      execSync(`curl -o "${cslPath}" "https://raw.githubusercontent.com/citation-style-language/styles/master/apa.csl"`);
      console.log('✅ APA CSL style downloaded');
    } catch (error) {
      console.error('❌ Failed to download APA CSL style:', error.message);
      process.exit(1);
    }
  }
}

async function processDirectory(dir, bibPath, cslPath) {
  const items = readdirSync(dir);
  
  for (const item of items) {
    const fullPath = join(dir, item);
    const stat = statSync(fullPath);
    
    if (stat.isDirectory()) {
      await processDirectory(fullPath, bibPath, cslPath);
    } else if (extname(item) === '.md') {
      await processMarkdownFile(fullPath, bibPath, cslPath);
    }
  }
}

async function processMarkdownFile(filePath, bibPath, cslPath) {
  try {
    console.log(`🔍 Processing ${filePath}...`);
    
    // Read and parse frontmatter
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data: frontmatter, content } = matter(fileContent);
    
    // Check if file has citations (Pandoc format: [@key] or @key)
    const hasCitations = /\[@\w+\]|@\w+/.test(content);
    
    if (!hasCitations) {
      console.log(`   ⏭️  No citations found, skipping`);
      return;
    }

    console.log(`   📖 Processing citations with Pandoc...`);
    
    // Create temporary files
    const tempDir = '/tmp/pandoc-preprocess';
    execSync(`mkdir -p "${tempDir}"`);
    
    const tempInput = join(tempDir, 'input.md');
    const tempOutput = join(tempDir, 'output.html');
    
    // Write content to temp file
    writeFileSync(tempInput, content);
    
    // Run Pandoc with citation processing
    const pandocCmd = [
      'pandoc',
      '--from', 'markdown',
      '--to', 'html',
      '--citeproc',
      '--bibliography', `"${bibPath}"`,
      '--csl', `"${cslPath}"`,
      `"${tempInput}"`,
      '-o', `"${tempOutput}"`
    ].join(' ');
    
    try {
      execSync(pandocCmd, { stdio: 'pipe' });
      
      // Read processed output
      const processedHtml = readFileSync(tempOutput, 'utf-8');
      
      // Convert back to markdown format
      const processedContent = convertPandocToMarkdown(processedHtml);
      
      // Combine with frontmatter and save
      const result = matter.stringify(processedContent, frontmatter);
      writeFileSync(filePath, result);
      
      console.log(`   ✅ Citations processed and saved`);
      
    } catch (pandocError) {
      console.error(`   ❌ Pandoc error: ${pandocError.message}`);
    }
    
    // Clean up
    execSync(`rm -f "${tempInput}" "${tempOutput}"`);
    
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
  }
}

function convertPandocToMarkdown(html) {
  let result = html;
  
  // First, collect all citations to number them
  const citations = [];
  const citationMatches = html.matchAll(/<span\s+class="citation"\s+data-cites="([^"]+)">([^<]+)<\/span>/gs);
  
  for (const match of citationMatches) {
    const citeKey = match[1];
    if (!citations.find(c => c.key === citeKey)) {
      citations.push({ key: citeKey, number: citations.length + 1 });
    }
  }
  
  // Convert Pandoc citations to numbered clickable links
  result = result.replace(
    /<span\s+class="citation"\s+data-cites="([^"]+)">([^<]+)<\/span>/gs,
    (match, citeKey, text) => {
      const citation = citations.find(c => c.key === citeKey);
      const number = citation ? citation.number : 1;
      return `[**[${number}]**](#ref-${citeKey})`;
    }
  );
  
  // Process bibliography - create markdown-safe version
  const bibliographyMatch = result.match(/<div id="refs" class="references[^>]*>.*?<\/div>/s);
  if (bibliographyMatch) {
    console.log('   📚 Found bibliography section, converting...');
    const bibliography = formatBibliographyAsMarkdown(bibliographyMatch[0]);
    result = result.replace(bibliographyMatch[0], bibliography);
  } else {
    console.log('   ⚠️  No bibliography section found in Pandoc output');
  }
  
  // Convert paragraphs back to markdown
  result = result.replace(/<p>/g, '').replace(/<\/p>/g, '\n\n');
  
  // Remove divs
  result = result.replace(/<\/?div[^>]*>/g, '');
  
  return result.trim();
}

function formatBibliographyAsMarkdown(bibliographyHtml) {
  // Extract individual references from Pandoc output
  const references = bibliographyHtml.match(/<div[^>]*class="csl-entry"[^>]*>.*?<\/div>/gs) || [];
  
  console.log(`   📖 Found ${references.length} bibliography entries`);
  
  let formattedBib = '\n\n## References\n\n';
  
  references.forEach((ref, index) => {
    const number = index + 1;
    
    // Extract the content inside the csl-entry div and clean it
    const contentMatch = ref.match(/<div[^>]*class="csl-entry"[^>]*>(.*?)<\/div>/s);
    let cleanRef = contentMatch ? contentMatch[1].trim() : ref.replace(/<\/?div[^>]*>/g, '').trim();
    
    // Extract the reference ID from the id attribute
    const refId = extractReferenceId(ref, index);
    
    // Convert HTML to markdown-friendly format (handle multiline)
    cleanRef = cleanRef.replace(/<em>/g, '*').replace(/<\/em>/g, '*');
    cleanRef = cleanRef.replace(/<span[^>]*>/g, '').replace(/<\/span>/g, '');
    cleanRef = cleanRef.replace(/<a\s+href="([^"]+)"[^>]*>([^<]+)<\/a>/gs, '[$2]($1)');
    
    // Clean up extra whitespace and format with proper indentation
    cleanRef = cleanRef.replace(/\s+/g, ' ').trim();
    
    // Add the reference using pure markdown format with anchor and proper spacing
    formattedBib += `<span id="ref-${refId}"></span>\n\n${number}. ${cleanRef}\n\n`;
  });
  
  return formattedBib;
}

function extractReferenceId(refHtml, fallbackIndex) {
  const match = refHtml.match(/id="ref-([^"]+)"/);
  return match ? match[1] : `item-${fallbackIndex + 1}`;
}

// Run the script
main().catch(error => {
  console.error('❌ Citation preprocessing failed:', error);
  process.exit(1);
});