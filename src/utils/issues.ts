import type { CollectionEntry } from 'astro:content';
import { buildPostCatalog, type PostCatalogItem } from './postCatalog';
import { filterPublishedPosts } from './postVisibility';

export type IssueEntry = CollectionEntry<'issues'>;
export type IssueReference = {
  slug: string;
  title: string;
  pubDate: Date;
};

export type ResolvedIssueItem = {
  post: PostCatalogItem;
  note: string | null;
  featured: boolean;
};

export type IssuePostContext = {
  catalog: PostCatalogItem[];
  allPostsBySlug: Map<string, CollectionEntry<'posts'>>;
  publishedPostsBySlug: Map<string, PostCatalogItem>;
};

const ISSUE_TITLE_PREFIX_PATTERN = /^issue\s*#?\s*\d+\s*[:\-]\s*/i;
const ISSUE_TITLE_NUMBER_PATTERN = /issue\s*#?\s*(\d+)/i;
const ISSUE_SLUG_NUMBER_PATTERN = /(?:^|-)issue-(\d+)(?:-|$)/i;

export const filterPublishedIssues = (issues: IssueEntry[]) =>
  issues.filter((issue) => !issue.data.draft);

export const sortIssuesByPubDateDesc = (issues: IssueEntry[]) =>
  [...issues].sort((left, right) => right.data.pubDate.getTime() - left.data.pubDate.getTime());

export const toIssueDisplayTitle = (title: string) => {
  const trimmed = title.trim();
  if (!trimmed) return title;
  const withoutPrefix = trimmed.replace(ISSUE_TITLE_PREFIX_PATTERN, '').trim();
  return withoutPrefix || trimmed;
};

export const toIssueEditionLabel = (title: string, slug = '') => {
  const titleNumber = title.match(ISSUE_TITLE_NUMBER_PATTERN)?.[1];
  const slugNumber = slug.match(ISSUE_SLUG_NUMBER_PATTERN)?.[1];
  const issueNumber = titleNumber ?? slugNumber;
  return issueNumber ? `Edition ${issueNumber}` : 'Edition';
};

export const buildIssuePostContext = (entries: CollectionEntry<'posts'>[]): IssuePostContext => {
  const publishedEntries = filterPublishedPosts(entries);
  const catalog = buildPostCatalog(publishedEntries);
  const allPostsBySlug = new Map(entries.map((entry) => [entry.slug, entry]));
  const publishedPostsBySlug = new Map(catalog.map((item) => [item.slug, item]));

  return {
    catalog,
    allPostsBySlug,
    publishedPostsBySlug,
  };
};

export const validateIssueReferences = (issue: IssueEntry, context: IssuePostContext) => {
  const referencedSlugs = Array.from(
    new Set(issue.data.items.map((item) => item.slug.trim()).filter(Boolean))
  );

  const missingSlugs = referencedSlugs.filter((slug) => !context.allPostsBySlug.has(slug));
  const unpublishedSlugs = referencedSlugs.filter(
    (slug) => context.allPostsBySlug.has(slug) && !context.publishedPostsBySlug.has(slug)
  );

  if (missingSlugs.length === 0 && unpublishedSlugs.length === 0) return;

  const lines = [`Issue "${issue.slug}" has invalid post references.`];

  if (missingSlugs.length > 0) {
    lines.push(`Missing post slug(s): ${missingSlugs.join(', ')}`);
  }

  if (unpublishedSlugs.length > 0) {
    lines.push(`Unpublished post slug(s): ${unpublishedSlugs.join(', ')}`);
  }

  throw new Error(lines.join('\n'));
};

export const resolveIssueItems = (issue: IssueEntry, context: IssuePostContext): ResolvedIssueItem[] => {
  validateIssueReferences(issue, context);

  return issue.data.items.map((item, index) => {
    const slug = item.slug.trim();
    const post = context.publishedPostsBySlug.get(slug);
    if (!post) {
      throw new Error(
        `Issue "${issue.slug}" could not resolve post slug "${slug}" at items[${index}]`
      );
    }

    return {
      post,
      note: item.note?.trim() || null,
      featured: Boolean(item.featured),
    };
  });
};

export const sortResolvedIssueItems = (
  items: ResolvedIssueItem[],
  sortBy: IssueEntry['data']['sortBy'],
  sortDirection: IssueEntry['data']['sortDirection']
) => {
  const direction = sortDirection === 'desc' ? -1 : 1;

  return [...items].sort((left, right) => {
    const leftDate = sortBy === 'updated' ? left.post.updatedDate : left.post.publishedDate;
    const rightDate = sortBy === 'updated' ? right.post.updatedDate : right.post.publishedDate;
    const delta = leftDate.getTime() - rightDate.getTime();

    if (delta !== 0) return delta * direction;
    return left.post.title.localeCompare(right.post.title) * direction;
  });
};

export const buildIssueReverseIndex = (issues: IssueEntry[]) => {
  const reverseMap = new Map<string, IssueReference[]>();

  for (const issue of issues) {
    const issueRef: IssueReference = {
      slug: issue.slug,
      title: issue.data.title,
      pubDate: issue.data.pubDate,
    };
    const seen = new Set<string>();

    for (const item of issue.data.items) {
      const postSlug = item.slug.trim();
      if (!postSlug || seen.has(postSlug)) continue;
      seen.add(postSlug);

      const current = reverseMap.get(postSlug) ?? [];
      current.push(issueRef);
      reverseMap.set(postSlug, current);
    }
  }

  reverseMap.forEach((issueRefs, slug) => {
    reverseMap.set(
      slug,
      [...issueRefs].sort((left, right) => {
        const dateDelta = right.pubDate.getTime() - left.pubDate.getTime();
        if (dateDelta !== 0) return dateDelta;
        return left.title.localeCompare(right.title);
      })
    );
  });

  return reverseMap;
};
