import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

export interface GitDates {
  created: Date;
  modified: Date;
}

// Build-time cache file
const CACHE_FILE = '.astro/git-dates-cache.json';

/**
 * Ultra-optimized build-time git dates generation
 */
export function generateGitDatesCache(): void {
  const cache: Record<string, GitDates | null> = {};
  
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
    const fileHistory: Record<string, { created: string; modified: string }> = {};
    
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

/**
 * Runtime git dates lookup - reads from pre-generated cache
 */
export function getGitDatesFromCache(filePath: string): GitDates | null {
  try {
    const cacheFile = join(process.cwd(), CACHE_FILE);
    if (!existsSync(cacheFile)) {
      return null;
    }
    
    const cache = JSON.parse(readFileSync(cacheFile, 'utf8'));
    const entry = cache[filePath];
    
    if (entry) {
      return {
        created: new Date(entry.created),
        modified: new Date(entry.modified)
      };
    }
    
    return null;
  } catch (error) {
    console.warn(`Failed to read git dates cache for ${filePath}:`, error);
    return null;
  }
}

/**
 * Enhanced getGitDatesForContent that uses cache in production
 */
export async function getGitDatesForContent(collectionName: string, slug: string): Promise<GitDates | null> {
  const filePath = `src/content/${collectionName}/${slug}.md`;
  
  // In production/build, use cache
  if (process.env.NODE_ENV === 'production' || import.meta.env.PROD) {
    let dates = getGitDatesFromCache(filePath);
    
    // Try original path for moved files
    if (!dates && (slug.startsWith('en/') || slug.startsWith('ja/'))) {
      const originalSlug = slug.replace(/^(en|ja)\//, '');
      const originalFilePath = `src/content/${collectionName}/${originalSlug}.md`;
      dates = getGitDatesFromCache(originalFilePath);
    }
    
    return dates;
  }
  
  // In development, fall back to original implementation
  const { getGitDates } = await import('./gitDates');
  return getGitDates(filePath);
}
