#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function showUsage() {
  console.log(`
Usage: node translate.js <post-filename>

Example:
  node translate.js 2023-08-15-python-for-data-science.md

This will:
1. Read the English post from src/content/posts/en/
2. Display the content for you to translate with Claude
3. Create a Japanese version in src/content/posts/ja/

Workflow:
1. Run this script with your post filename
2. Copy the English content shown
3. Paste to Claude with: "Translate this markdown blog post to Japanese, keeping all frontmatter and code blocks intact:"
4. Paste Claude's translation back when prompted
`);
}

function main() {
  const filename = process.argv[2];
  
  if (!filename) {
    showUsage();
    process.exit(1);
  }

  const enPath = path.join(__dirname, 'src/content/posts/en', filename);
  const jaPath = path.join(__dirname, 'src/content/posts/ja', filename);

  if (!fs.existsSync(enPath)) {
    console.error(`❌ File not found: ${enPath}`);
    process.exit(1);
  }

  if (fs.existsSync(jaPath)) {
    console.log(`⚠️  Japanese version already exists: ${jaPath}`);
    console.log('Overwrite? (y/N):');
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on('data', (key) => {
      if (key.toString() === 'y' || key.toString() === 'Y') {
        startTranslation();
      } else {
        console.log('\n❌ Translation cancelled');
        process.exit(0);
      }
    });
    return;
  }

  startTranslation();

  function startTranslation() {
    const content = fs.readFileSync(enPath, 'utf8');
    
    console.log(`\n📝 English content from ${filename}:`);
    console.log('=' + '='.repeat(50));
    console.log(content);
    console.log('=' + '='.repeat(50));
    
    console.log(`\n🤖 Copy the above content and paste to Claude with this prompt:`);
    console.log(`"Translate this markdown blog post to Japanese, keeping all frontmatter and code blocks intact:"`);
    
    console.log(`\n📥 Paste Claude's Japanese translation below (press Ctrl+D when done):`);
    
    let translation = '';
    process.stdin.setEncoding('utf8');
    process.stdin.setRawMode(false);
    process.stdin.resume();
    
    process.stdin.on('data', (chunk) => {
      translation += chunk;
    });
    
    process.stdin.on('end', () => {
      if (translation.trim()) {
        fs.writeFileSync(jaPath, translation.trim());
        console.log(`\n✅ Japanese version created: ${jaPath}`);
        console.log(`\n🌐 URLs will be:`);
        console.log(`   English: /${filename.replace('.md', '')}`);
        console.log(`   Japanese: /ja/${filename.replace('.md', '')}`);
      } else {
        console.log('\n❌ No translation provided');
      }
      process.exit(0);
    });
  }
}

main();