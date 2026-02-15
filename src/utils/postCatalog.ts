import type { CollectionEntry } from 'astro:content';
import { getPostDates } from './postDates';
import { normalizeTag } from './tags';
import { hasCitationContent, hasMathContent } from './contentFeatures';
import { extractCitationKeys } from './citationKeys';

export type PostLabel = {
  label: string;
  slug: string;
};

export type PostCatalogItem = {
  entry: CollectionEntry<'posts'>;
  slug: string;
  title: string;
  abstract: string;
  tags: PostLabel[];
  topics: PostLabel[];
  normalizedTags: string;
  normalizedTopics: string;
  publishedDate: Date;
  updatedDate: Date;
  pinned: boolean;
  hasMath: boolean;
  hasCitations: boolean;
  citationKeys: string[];
  difficulty: 'intro' | 'intermediate' | 'advanced' | null;
  status: 'reviewed' | 'evergreen' | 'archived' | null;
};

const byPinnedThenUpdated = (a: PostCatalogItem, b: PostCatalogItem) => {
  if (a.pinned !== b.pinned) {
    return a.pinned ? -1 : 1;
  }
  return b.updatedDate.getTime() - a.updatedDate.getTime();
};

const normalizeLabels = (values: string[] = []): PostLabel[] => {
  const seen = new Set<string>();
  const labels: PostLabel[] = [];

  for (const rawValue of values) {
    const slug = normalizeTag(rawValue);
    if (!slug || seen.has(slug)) continue;
    seen.add(slug);
    labels.push({ label: rawValue, slug });
  }

  return labels;
};

export function buildPostCatalog(entries: CollectionEntry<'posts'>[]): PostCatalogItem[] {
  const catalog = entries.map((entry) => {
    const tags = normalizeLabels(entry.data.tags ?? []);
    const topics = normalizeLabels(entry.data.topics ?? []);

    const { published, updated } = getPostDates(entry);
    const abstract = (entry.data.description || entry.data.summary || '').trim();

    return {
      entry,
      slug: entry.slug,
      title: entry.data.title,
      abstract,
      tags,
      topics,
      normalizedTags: tags.map((tag) => tag.slug).join(' '),
      normalizedTopics: topics.map((topic) => topic.slug).join(' '),
      publishedDate: published,
      updatedDate: updated,
      pinned: Boolean(entry.data.pinned),
      hasMath: hasMathContent(entry.body),
      hasCitations: hasCitationContent(entry.body),
      citationKeys: extractCitationKeys(entry.body),
      difficulty: entry.data.difficulty ?? null,
      status: entry.data.status ?? null,
    };
  });

  return catalog.sort(byPinnedThenUpdated);
}
