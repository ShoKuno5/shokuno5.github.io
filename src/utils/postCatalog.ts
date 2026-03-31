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
  entry: CollectionEntry<'writing'>;
  slug: string;
  title: string;
  abstract: string;
  tags: PostLabel[];
  normalizedTags: string;
  publishedDate: Date;
  updatedDate: Date;
  hasMath: boolean;
  hasCitations: boolean;
  citationKeys: string[];
  maturity: 'seedling' | 'budding' | 'evergreen';
};

const byUpdated = (a: PostCatalogItem, b: PostCatalogItem) =>
  b.updatedDate.getTime() - a.updatedDate.getTime();

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

export function buildPostCatalog(entries: CollectionEntry<'writing'>[]): PostCatalogItem[] {
  const catalog = entries.map((entry) => {
    const tags = normalizeLabels(entry.data.tags ?? []);
    const { published, updated } = getPostDates(entry);
    const abstract = (entry.data.description || entry.data.summary || '').trim();

    return {
      entry,
      slug: entry.slug,
      title: entry.data.title,
      abstract,
      tags,
      normalizedTags: tags.map((tag) => tag.slug).join(' '),
      publishedDate: published,
      updatedDate: updated,
      hasMath: hasMathContent(entry.body),
      hasCitations: hasCitationContent(entry.body),
      citationKeys: extractCitationKeys(entry.body),
      maturity: entry.data.maturity ?? 'seedling',
    };
  });

  return catalog.sort(byUpdated);
}
