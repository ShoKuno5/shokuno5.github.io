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
      author: z.string().optional(),
      summary: z.string().optional(),
      pinned: z.boolean().optional(),
      heroImage: image().optional(),
      type: z.string().optional(),
    }),
});

export const collections = {
  posts,
};
