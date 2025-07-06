import { execSync } from 'child_process';
import { existsSync } from 'fs';

export interface GitDates {
  created: Date;
  modified: Date;
}

// Cache to avoid repeated git calls
const gitDatesCache = new Map<string, GitDates | null>();

/**
 * Get git dates for a file
 * @param filePath - Absolute path to the file
 * @returns Object with created and modified dates
 */
export function getGitDates(filePath: string): GitDates | null {
  // Check cache first
  if (gitDatesCache.has(filePath)) {
    return gitDatesCache.get(filePath)!;
  }

  try {
    // Check if file exists
    if (!existsSync(filePath)) {
      gitDatesCache.set(filePath, null);
      return null;
    }

    // Use a single git command to get both dates more efficiently
    const gitCommand = `git log --follow --format="%ai" --reverse -- "${filePath}"`;
    const gitOutput = execSync(gitCommand, { 
      encoding: 'utf8', 
      cwd: process.cwd(),
      timeout: 5000 // 5 second timeout
    }).trim();
    
    if (!gitOutput) {
      gitDatesCache.set(filePath, null);
      return null;
    }

    const lines = gitOutput.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
      gitDatesCache.set(filePath, null);
      return null;
    }

    const created = new Date(lines[0]); // First commit (oldest)
    const modified = new Date(lines[lines.length - 1]); // Last commit (newest)

    const result = { created, modified };
    gitDatesCache.set(filePath, result);
    return result;
  } catch (error) {
    console.warn(`Failed to get git dates for ${filePath}:`, error);
    gitDatesCache.set(filePath, null);
    return null;
  }
}

/**
 * Get git dates for a content collection entry
 * @param collectionName - Name of the collection (e.g., 'posts')
 * @param slug - The entry slug (may include language prefix like 'en/post-name')
 * @returns Object with created and modified dates
 */
export function getGitDatesForContent(collectionName: string, slug: string): GitDates | null {
  const filePath = `src/content/${collectionName}/${slug}.md`;
  
  // First try the current path
  let gitDates = getGitDates(filePath);
  
  // If no git dates found and the slug has a language prefix, try the original flat structure
  // This handles the case where files were moved from flat structure to en/ja folders
  if (!gitDates && (slug.startsWith('en/') || slug.startsWith('ja/'))) {
    const originalSlug = slug.replace(/^(en|ja)\//, '');
    const originalFilePath = `src/content/${collectionName}/${originalSlug}.md`;
    gitDates = getGitDatesFromPath(originalFilePath);
  }
  
  return gitDates;
}

/**
 * Get git dates for a file path without checking if file exists
 * This is useful for getting git history of moved files
 */
function getGitDatesFromPath(filePath: string): GitDates | null {
  // Check cache first
  if (gitDatesCache.has(filePath)) {
    return gitDatesCache.get(filePath)!;
  }

  try {
    // Use a single git command to get both dates more efficiently
    const gitCommand = `git log --follow --format="%ai" --reverse -- "${filePath}"`;
    const gitOutput = execSync(gitCommand, { 
      encoding: 'utf8', 
      cwd: process.cwd(),
      timeout: 5000 // 5 second timeout
    }).trim();
    
    if (!gitOutput) {
      gitDatesCache.set(filePath, null);
      return null;
    }

    const lines = gitOutput.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
      gitDatesCache.set(filePath, null);
      return null;
    }

    const created = new Date(lines[0]); // First commit (oldest)
    const modified = new Date(lines[lines.length - 1]); // Last commit (newest)

    const result = { created, modified };
    gitDatesCache.set(filePath, result);
    return result;
  } catch (error) {
    console.warn(`Failed to get git dates for ${filePath}:`, error);
    gitDatesCache.set(filePath, null);
    return null;
  }
}