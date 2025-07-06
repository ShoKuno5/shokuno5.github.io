import { getCollection } from 'astro:content';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { getGitDatesForContent } from './gitDatesOptimized.js';
import { extractDateFromFilename } from './extractDateFromFilename.js';

/**
 * Generate static data for ultra-fast page loads
 */
export async function generateStaticData() {
  console.log('📊 Generating static data...');
  
  try {
    const posts = await getCollection('posts', ({ id }) => id.startsWith('en/') || !id.includes('/'));
    const visiblePosts = posts.filter(post => !post.data.hidden);
    
    // Process all posts with optimized data structure
    const processedPosts = await Promise.all(
      visiblePosts.map(async (post) => {
        const gitDates = await getGitDatesForContent('posts', post.slug);
        const filenameDate = extractDateFromFilename(post.slug);
        const publishDate = post.data.date || filenameDate || gitDates?.created || new Date(0);
        
        let modifiedDate = post.data.modified;
        if (!modifiedDate && gitDates?.modified) {
          if (gitDates.modified.getTime() !== publishDate.getTime()) {
            modifiedDate = gitDates.modified;
          }
        }
        
        return {
          slug: post.slug,
          title: post.data.title,
          summary: post.data.summary,
          description: post.data.description,
          tags: post.data.tags || [],
          publishDate: publishDate.toISOString(),
          modifiedDate: modifiedDate?.toISOString() || null,
          wordCount: post.body.split(/\s+/).length
        };
      })
    );
    
    // Generate tag statistics
    const tagStats = new Map<string, { count: number; posts: string[] }>();
    processedPosts.forEach(post => {
      post.tags.forEach(tag => {
        const normalizedTag = tag.toLowerCase();
        if (!tagStats.has(normalizedTag)) {
          tagStats.set(normalizedTag, { count: 0, posts: [] });
        }
        const stats = tagStats.get(normalizedTag)!;
        stats.count++;
        stats.posts.push(post.slug);
      });
    });
    
    const staticData = {
      posts: processedPosts,
      tags: Object.fromEntries(tagStats.entries()),
      totalPosts: processedPosts.length,
      totalTags: tagStats.size,
      generated: new Date().toISOString()
    };
    
    // Ensure cache directory exists
    const cacheDir = join(process.cwd(), '.astro');
    try {
      mkdirSync(cacheDir, { recursive: true });
    } catch (error) {
      // Directory might already exist
    }
    
    writeFileSync(
      join(process.cwd(), '.astro/static-data.json'),
      JSON.stringify(staticData, null, 2)
    );
    
    console.log(`✅ Static data generated: ${processedPosts.length} posts, ${tagStats.size} tags`);
  } catch (error) {
    console.error('❌ Failed to generate static data:', error);
  }
}
