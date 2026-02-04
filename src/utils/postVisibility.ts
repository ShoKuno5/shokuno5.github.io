import type { CollectionEntry } from 'astro:content';
import { SITE } from '../config/site.js';
import { normalizeTag } from './tags';

const defaultExcludeTags = ['draft', 'private'];

const configuredTags = SITE.posts?.excludeTags ?? defaultExcludeTags;
const excludedTags = configuredTags
  .map((tag) => normalizeTag(tag))
  .filter(Boolean);

const hasExcludedTag = (tags: string[] = []) => {
  if (excludedTags.length === 0) return false;
  return tags.map((tag) => normalizeTag(tag)).some((tag) => excludedTags.includes(tag));
};

export const shouldPublishPost = (entry: CollectionEntry<'posts'>) =>
  !hasExcludedTag(entry.data.tags ?? []);

export const filterPublishedPosts = (entries: CollectionEntry<'posts'>[]) =>
  entries.filter(shouldPublishPost);
