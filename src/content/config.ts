import { z, defineCollection } from 'astro:content';

const staticPageSchema = z.object({
  title: z.string(),
  description: z.string(),
});

export const collections = {
  posts: defineCollection({
    type: 'content',
    schema: z.object({
      title: z.string(),
      date: z.date(),
      tags: z.array(z.string()).optional(),
      summary: z.string().optional(),
      hidden: z.boolean().optional(),
    }),
  }),
  projects: defineCollection({
    type: 'content',
    schema: z.object({
      title: z.string(),
      description: z.string(),
      link: z.string().url(),
      technologies: z.array(z.string()),
      order: z.number(),
    }),
  }),
  'naive-hope': defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
};