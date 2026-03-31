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

const visual = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string().optional(),
      date: z.coerce.date(),
      medium: z.enum(['photography', '3d', 'film']).default('photography'),
      coverImage: image(),
      tags: z.array(z.string()).default([]),
      featured: z.boolean().default(false),
    }),
});

export const collections = { writing, visual };
