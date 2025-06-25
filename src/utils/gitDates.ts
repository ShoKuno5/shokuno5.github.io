import { execSync } from 'child_process';
import { existsSync } from 'fs';

export interface GitDates {
  created: Date;
  modified: Date;
}

/**
 * Get git dates for a file
 * @param filePath - Absolute path to the file
 * @returns Object with created and modified dates
 */
export function getGitDates(filePath: string): GitDates | null {
  try {
    // Check if file exists
    if (!existsSync(filePath)) {
      return null;
    }

    // Get first commit date (creation)
    const createdCommand = `git log --follow --format=%ai --diff-filter=A -- "${filePath}" | tail -1`;
    const createdOutput = execSync(createdCommand, { encoding: 'utf8', cwd: process.cwd() }).trim();
    
    // Get last commit date (modification)
    const modifiedCommand = `git log -1 --format=%ai -- "${filePath}"`;
    const modifiedOutput = execSync(modifiedCommand, { encoding: 'utf8', cwd: process.cwd() }).trim();
    
    if (!createdOutput || !modifiedOutput) {
      return null;
    }

    return {
      created: new Date(createdOutput),
      modified: new Date(modifiedOutput)
    };
  } catch (error) {
    console.warn(`Failed to get git dates for ${filePath}:`, error);
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
  try {
    // Get first commit date (creation) - don't check if file exists
    const createdCommand = `git log --follow --format=%ai --diff-filter=A -- "${filePath}" | tail -1`;
    const createdOutput = execSync(createdCommand, { encoding: 'utf8', cwd: process.cwd() }).trim();
    
    // Get last commit date (modification)
    const modifiedCommand = `git log -1 --format=%ai -- "${filePath}"`;
    const modifiedOutput = execSync(modifiedCommand, { encoding: 'utf8', cwd: process.cwd() }).trim();
    
    if (!createdOutput || !modifiedOutput) {
      return null;
    }

    return {
      created: new Date(createdOutput),
      modified: new Date(modifiedOutput)
    };
  } catch (error) {
    console.warn(`Failed to get git dates for ${filePath}:`, error);
    return null;
  }
}