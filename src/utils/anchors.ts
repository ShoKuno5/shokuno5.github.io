const unsafeAnchorChars = /[^a-zA-Z0-9_-]+/g;

export const slugToAnchorId = (slug: string) => `post-${slug.replace(unsafeAnchorChars, '-')}`;
