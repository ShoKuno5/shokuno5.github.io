import type { PostCatalogItem } from './postCatalog';

export type CitedByPost = {
  slug: string;
  title: string;
  sharedKeys: string[];
  updatedDate: Date;
};

export type CitationIndex = {
  keyToSlugs: Map<string, Set<string>>;
  slugToKeys: Map<string, string[]>;
};

export function buildCitationIndex(posts: PostCatalogItem[]): CitationIndex {
  const keyToSlugs = new Map<string, Set<string>>();
  const slugToKeys = new Map<string, string[]>();

  posts.forEach((post) => {
    slugToKeys.set(post.slug, post.citationKeys);

    post.citationKeys.forEach((key) => {
      if (!keyToSlugs.has(key)) {
        keyToSlugs.set(key, new Set<string>());
      }
      keyToSlugs.get(key)?.add(post.slug);
    });
  });

  return { keyToSlugs, slugToKeys };
}

export function getCitedByPosts(
  currentSlug: string,
  posts: PostCatalogItem[],
  index: CitationIndex,
  limit: number = 5
): CitedByPost[] {
  const currentKeys = index.slugToKeys.get(currentSlug) || [];
  if (currentKeys.length === 0) return [];

  const sharedBySlug = new Map<string, string[]>();

  currentKeys.forEach((key) => {
    const slugs = index.keyToSlugs.get(key);
    if (!slugs) return;

    slugs.forEach((slug) => {
      if (slug === currentSlug) return;
      if (!sharedBySlug.has(slug)) {
        sharedBySlug.set(slug, []);
      }
      sharedBySlug.get(slug)?.push(key);
    });
  });

  const bySlug = new Map(posts.map((post) => [post.slug, post]));
  const results: CitedByPost[] = [];

  sharedBySlug.forEach((sharedKeys, slug) => {
    const post = bySlug.get(slug);
    if (!post) return;

    results.push({
      slug: post.slug,
      title: post.title,
      sharedKeys: [...new Set(sharedKeys)],
      updatedDate: post.updatedDate,
    });
  });

  return results
    .sort((a, b) => {
      if (b.sharedKeys.length !== a.sharedKeys.length) {
        return b.sharedKeys.length - a.sharedKeys.length;
      }
      return b.updatedDate.getTime() - a.updatedDate.getTime();
    })
    .slice(0, limit);
}
