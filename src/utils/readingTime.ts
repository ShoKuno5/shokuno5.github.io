/**
 * Calculate reading time based on word count
 * @param content - The markdown content to analyze
 * @param wordsPerMinute - Average reading speed (default: 225 words/minute)
 * @returns Object with reading time in minutes and word count
 */
export function calculateReadingTime(content: string, wordsPerMinute: number = 225) {
  // Remove markdown syntax, HTML tags, and extra whitespace
  const cleanContent = content
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]*`/g, '') // Remove inline code
    .replace(/!\[.*?\]\(.*?\)/g, '') // Remove images
    .replace(/\[.*?\]\(.*?\)/g, '') // Remove links (keep text)
    .replace(/#+ /g, '') // Remove headers
    .replace(/[*_]{1,2}(.*?)[*_]{1,2}/g, '$1') // Remove bold/italic
    .replace(/^\s*[-*+]\s+/gm, '') // Remove list markers
    .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered list markers
    .replace(/^\s*>\s+/gm, '') // Remove blockquote markers
    .replace(/\n+/g, ' ') // Replace newlines with spaces
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();

  // Count words (split by whitespace and filter empty strings)
  const wordCount = cleanContent
    .split(/\s+/)
    .filter(word => word.length > 0).length;

  // Calculate reading time in minutes
  const readingTimeMinutes = Math.ceil(wordCount / wordsPerMinute);

  return {
    wordCount,
    readingTime: readingTimeMinutes,
    readingTimeText: formatReadingTime(readingTimeMinutes)
  };
}

/**
 * Format reading time into human-readable text
 */
function formatReadingTime(minutes: number): string {
  if (minutes < 1) {
    return 'Less than 1 min read';
  } else if (minutes === 1) {
    return '1 min read';
  } else {
    return `${minutes} min read`;
  }
}

/**
 * Get word count from rendered content (for more accurate counting)
 */
export function getWordCountFromRendered(renderedContent: string): number {
  // Remove HTML tags and get plain text
  const textContent = renderedContent
    .replace(/<[^>]*>/g, '') // Remove HTML tags
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();

  return textContent
    .split(/\s+/)
    .filter(word => word.length > 0).length;
}