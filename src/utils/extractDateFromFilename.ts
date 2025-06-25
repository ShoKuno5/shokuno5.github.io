/**
 * Extract date from filename pattern YYYY-MM-DD-title.md
 * @param slug - The post slug (filename without extension)
 * @returns Date object or null if no date found
 */
export function extractDateFromFilename(slug: string): Date | null {
  // Match pattern: YYYY-MM-DD at the start of the slug
  const dateMatch = slug.match(/^(\d{4})-(\d{2})-(\d{2})-/);
  
  if (dateMatch) {
    const [, year, month, day] = dateMatch;
    const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
    
    // Validate the date is reasonable (not in the future, not before 2000)
    if (date.getTime() > 0 && date.getFullYear() >= 2000 && date <= new Date()) {
      return date;
    }
  }
  
  return null;
}