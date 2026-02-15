import { z, defineCollection } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string().optional(),
      pubDate: z.coerce.date().optional(),
      modified: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      topics: z.array(z.string()).default([]),
      author: z.string().optional(),
      summary: z.string().optional(),
      pinned: z.boolean().optional(),
      difficulty: z.enum(['intro', 'intermediate', 'advanced']).optional(),
      status: z.enum(['reviewed', 'evergreen', 'archived']).optional(),
      heroImage: image().optional(),
      type: z.string().optional(),
    }),
});

const issues = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string().optional(),
      pubDate: z.coerce.date(),
      draft: z.boolean().default(false),
      coverImage: image().optional(),
      sortBy: z.enum(['published', 'updated']).default('published'),
      sortDirection: z.enum(['asc', 'desc']).default('asc'),
      items: z.array(
        z.object({
          slug: z.string(),
          note: z.string().optional(),
          featured: z.boolean().optional(),
        })
      ),
    }),
});

export const collections = {
  posts,
  issues,
};
