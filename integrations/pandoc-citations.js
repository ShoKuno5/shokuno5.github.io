import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';
import { execSync } from 'child_process';
import matter from 'gray-matter';

export default function pandocCitations() {
  return {
    name: 'pandoc-citations',
    hooks: {
      'astro:config:setup': () => {
        console.log('🔬 Pandoc Citations integration initialized');
      },
      'astro:build:start': async ({ config }) => {
        console.log('📚 Processing citations with Pandoc...');
        await processCitationsInContent(config.root);
      }
    }
  };
}

async function processCitationsInContent(projectRoot) {
  const contentDir = join(projectRoot.pathname, 'src/content/posts');
  const bibPath = join(projectRoot.pathname, 'src/content/bibliography.bib');
  const cslPath = join(projectRoot.pathname, 'scripts/apa.csl');
  
  // Ensure APA CSL style exists
  await ensureCSLStyle(cslPath);
  
  // Process all markdown files in content directories
  await processDirectory(contentDir, bibPath, cslPath);
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
      throw error;
    }
  }
}

async function processDirectory(dir, bibPath, cslPath) {
  try {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Recursively process subdirectories
        await processDirectory(fullPath, bibPath, cslPath);
      } else if (extname(item) === '.md') {
        await processMarkdownFile(fullPath, bibPath, cslPath);
      }
    }
  } catch (error) {
    console.error(`Error processing directory ${dir}:`, error.message);
  }
}

async function processMarkdownFile(filePath, bibPath, cslPath) {
  try {
    console.log(`🔍 Checking ${filePath}...`);
    
    // Read and parse frontmatter
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data: frontmatter, content } = matter(fileContent);
    
    // Check if file has citations
    const hasCitations = /\\cite\{|@\w+[{\s]/.test(content);
    
    if (!hasCitations) {
      console.log(`   ⏭️  No citations found, skipping`);
      return;
    }

    console.log(`   📖 Processing citations...`);
    
    // Convert LaTeX-style citations to Pandoc format
    let pandocContent = content;
    
    // Convert \cite{key} to [@key]
    pandocContent = pandocContent.replace(/\\cite\{([^}]+)\}/g, '[@$1]');
    
    // Create temporary files
    const tempDir = '/tmp/pandoc-astro';
    execSync(`mkdir -p "${tempDir}"`);
    
    const tempInput = join(tempDir, 'input.md');
    const tempOutput = join(tempDir, 'output.html');
    
    // Write content to temp file
    writeFileSync(tempInput, pandocContent);
    
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
      
      // Convert back to Astro-compatible format
      const processedContent = convertPandocToAstro(processedHtml);
      
      // Combine with frontmatter and save
      const result = matter.stringify(processedContent, frontmatter);
      writeFileSync(filePath, result);
      
      console.log(`   ✅ Citations processed successfully`);
      
    } catch (pandocError) {
      console.error(`   ❌ Pandoc error: ${pandocError.message}`);
      // Don't fail the build, just skip this file
    }
    
    // Clean up
    execSync(`rm -f "${tempInput}" "${tempOutput}"`);
    
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
  }
}

function convertPandocToAstro(html) {
  let result = html;
  
  // Convert Pandoc citations to clickable links
  result = result.replace(
    /<a href="#ref-([^"]+)" role="doc-biblioref">([^<]+)<\/a>/g,
    '<a href="#ref-$1" class="citation-link">[$2]</a>'
  );
  
  // Process bibliography
  const bibliographyMatch = result.match(/<div id="refs" class="references[^>]*>(.*?)<\/div>/s);
  if (bibliographyMatch) {
    const bibliography = formatBibliography(bibliographyMatch[1]);
    result = result.replace(bibliographyMatch[0], bibliography);
  }
  
  // Convert HTML elements back to markdown-compatible format
  result = result.replace(/<p>/g, '').replace(/<\/p>/g, '\n\n');
  result = result.replace(/<\/?div[^>]*>/g, '');
  
  return result.trim();
}

function formatBibliography(bibliographyHtml) {
  // Extract individual references
  const references = bibliographyHtml.match(/<div[^>]*class="csl-entry"[^>]*>.*?<\/div>/gs) || [];
  
  let formattedBib = '\n\n## References\n\n';
  formattedBib += '<div class="bibliography">\n';
  formattedBib += '<h2 class="bibliography-title">References</h2>\n';
  formattedBib += '<div class="bibliography-list">\n';
  
  references.forEach((ref, index) => {
    const number = index + 1;
    const cleanRef = ref.replace(/<\/?div[^>]*>/g, '').trim();
    const refId = extractReferenceId(ref, index);
    
    formattedBib += `<div class="bibliography-entry" id="ref-${refId}">\n`;
    formattedBib += `  <div class="citation-layout">\n`;
    formattedBib += `    <span class="citation-number">[${number}]</span>\n`;
    formattedBib += `    <div class="citation-content">${cleanRef}</div>\n`;
    formattedBib += `  </div>\n`;
    formattedBib += `</div>\n`;
  });
  
  formattedBib += '</div>\n</div>\n';
  
  return formattedBib;
}

function extractReferenceId(refHtml, fallbackIndex) {
  const match = refHtml.match(/id="ref-([^"]+)"/);
  return match ? match[1] : `item-${fallbackIndex + 1}`;
}