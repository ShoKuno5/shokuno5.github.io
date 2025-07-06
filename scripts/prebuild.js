#!/usr/bin/env node

import { execSync } from 'child_process';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

console.log('🚀 Running ultra-optimized prebuild...');

const startTime = Date.now();

try {
  // Step 1: Generate git dates cache with single git command
  console.log('📦 Generating git dates cache...');
  generateGitDatesCache();
  
  const endTime = Date.now();
  console.log(`✅ Ultra-optimized prebuild completed in ${endTime - startTime}ms`);
} catch (error) {
  console.error('❌ Prebuild failed:', error);
  process.exit(1);
}

/**
 * Ultra-optimized build-time git dates generation
 */
function generateGitDatesCache() {
  const cache = {};
  const CACHE_FILE = '.astro/git-dates-cache.json';
  
  try {
    // Single git command to get all file histories at once
    const gitCommand = `git log --name-only --pretty=format:"%H|%ai" --follow -- src/content/posts/`;
    const gitOutput = execSync(gitCommand, { 
      encoding: 'utf8', 
      timeout: 10000,
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    }).trim();
    
    if (!gitOutput) {
      console.log('No git history found');
      return;
    }

    const commits = gitOutput.split('\n\n').filter(Boolean);
    const fileHistory = {};
    
    console.log(`Processing ${commits.length} commits...`);
    
    // Process commits in reverse chronological order
    commits.reverse().forEach(commit => {
      const lines = commit.split('\n');
      if (lines.length < 2) return;
      
      const [hashAndDate, ...filePaths] = lines;
      const [, dateStr] = hashAndDate.split('|');
      
      filePaths.forEach(filePath => {
        if (filePath && filePath.includes('src/content/posts/') && filePath.endsWith('.md')) {
          if (!fileHistory[filePath]) {
            fileHistory[filePath] = { created: dateStr, modified: dateStr };
          } else {
            fileHistory[filePath].modified = dateStr;
          }
        }
      });
    });
    
    // Convert to cache format
    Object.entries(fileHistory).forEach(([filePath, dates]) => {
      cache[filePath] = {
        created: new Date(dates.created),
        modified: new Date(dates.modified)
      };
    });
    
    // Ensure cache directory exists
    const cacheDir = join(process.cwd(), '.astro');
    try {
      mkdirSync(cacheDir, { recursive: true });
    } catch (error) {
      // Directory might already exist
    }
    
    writeFileSync(join(process.cwd(), CACHE_FILE), JSON.stringify(cache, null, 2));
    console.log(`✅ Git dates cache generated with ${Object.keys(cache).length} files`);
  } catch (error) {
    console.error('❌ Failed to generate git dates cache:', error);
  }
}
