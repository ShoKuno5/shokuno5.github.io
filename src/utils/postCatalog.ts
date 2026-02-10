import type { CollectionEntry } from 'astro:content';
import { getPostDates } from './postDates';
import { normalizeTag } from './tags';
import { hasCitationContent, hasMathContent } from './contentFeatures';

export type PostTag = {
  label: string;
  slug: string;
};

export type PostCatalogItem = {
  entry: CollectionEntry<'posts'>;
  slug: string;
  title: string;
  description: string;
  summary: string;
  excerpt: string;
  tags: PostTag[];
  normalizedTags: string;
  publishedDate: Date;
  updatedDate: Date;
  pinned: boolean;
  hasMath: boolean;
  hasCitations: boolean;
};

const byPinnedThenUpdated = (a: PostCatalogItem, b: PostCatalogItem) => {
  if (a.pinned !== b.pinned) {
    return a.pinned ? -1 : 1;
  }
  return b.updatedDate.getTime() - a.updatedDate.getTime();
};

export function buildPostCatalog(entries: CollectionEntry<'posts'>[]): PostCatalogItem[] {
  const catalog = entries.map((entry) => {
    const rawTags = entry.data.tags ?? [];
    const seenTags = new Set<string>();
    const tags: PostTag[] = [];

    for (const rawTag of rawTags) {
      const slug = normalizeTag(rawTag);
      if (!slug || seenTags.has(slug)) continue;
      seenTags.add(slug);
      tags.push({ label: rawTag, slug });
    }

    const { published, updated } = getPostDates(entry);
    const description = entry.data.description || '';
    const summary = entry.data.summary || '';
    const excerpt = description || summary;

    return {
      entry,
      slug: entry.slug,
      title: entry.data.title,
      description,
      summary,
      excerpt,
      tags,
      normalizedTags: tags.map((tag) => tag.slug).join(' '),
      publishedDate: published,
      updatedDate: updated,
      pinned: Boolean(entry.data.pinned),
      hasMath: hasMathContent(entry.body),
      hasCitations: hasCitationContent(entry.body),
    };
  });

  return catalog.sort(byPinnedThenUpdated);
}
