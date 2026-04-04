import { z, defineCollection } from 'astro:content';

const writing = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string().optional(),
      pubDate: z.coerce.date().optional(),
      modified: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      author: z.string().optional(),
      summary: z.string().optional(),
      heroImage: image().optional(),
      maturity: z.enum(['seedling', 'budding', 'evergreen']).default('seedling'),
    }),
});

export const collections = { writing };
