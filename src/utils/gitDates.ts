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
 * @param slug - The entry slug
 * @returns Object with created and modified dates
 */
export function getGitDatesForContent(collectionName: string, slug: string): GitDates | null {
  const filePath = `src/content/${collectionName}/${slug}.md`;
  return getGitDates(filePath);
}