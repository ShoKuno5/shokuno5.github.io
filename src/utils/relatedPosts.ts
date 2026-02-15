import type { PostCatalogItem } from './postCatalog';

export type RelatedPost = {
  slug: string;
  title: string;
  score: number;
  reasons: string[];
};

const intersectionCount = (left: string[], right: string[]) => {
  if (left.length === 0 || right.length === 0) return 0;
  const rightSet = new Set(right);
  let count = 0;
  left.forEach((item) => {
    if (rightSet.has(item)) count += 1;
  });
  return count;
};

export function getRelatedPosts(
  current: PostCatalogItem,
  posts: PostCatalogItem[],
  limit: number = 4
): RelatedPost[] {
  const related: RelatedPost[] = [];

  posts.forEach((candidate) => {
    if (candidate.slug === current.slug) return;

    const sharedTags = intersectionCount(
      current.tags.map((tag) => tag.slug),
      candidate.tags.map((tag) => tag.slug)
    );

    const sharedTopics = intersectionCount(
      current.topics.map((topic) => topic.slug),
      candidate.topics.map((topic) => topic.slug)
    );

    const sharedCitations = intersectionCount(current.citationKeys, candidate.citationKeys);
    const score = sharedTags + sharedTopics * 2 + sharedCitations * 3;
    if (score <= 0) return;

    const reasons: string[] = [];
    if (sharedCitations > 0) {
      reasons.push(`${sharedCitations} shared citation${sharedCitations === 1 ? '' : 's'}`);
    }
    if (sharedTopics > 0) {
      reasons.push(`${sharedTopics} shared topic${sharedTopics === 1 ? '' : 's'}`);
    }
    if (sharedTags > 0) {
      reasons.push(`${sharedTags} shared tag${sharedTags === 1 ? '' : 's'}`);
    }

    related.push({
      slug: candidate.slug,
      title: candidate.title,
      score,
      reasons,
    });
  });

  return related
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.title.localeCompare(b.title);
    })
    .slice(0, limit);
}
