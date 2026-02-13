import { z, defineCollection } from 'astro:content';

const postSeriesSchema = z.object({
  name: z.string(),
  order: z.number().int().optional(),
});

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
      series: z.union([z.string(), postSeriesSchema]).optional(),
      status: z.enum(['reviewed', 'evergreen', 'archived']).optional(),
      heroImage: image().optional(),
      type: z.string().optional(),
    }),
});

export const collections = {
  posts,
};
