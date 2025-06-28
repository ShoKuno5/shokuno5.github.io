import { z, defineCollection } from 'astro:content';

const staticPageSchema = z.object({
  title: z.string(),
  description: z.string(),
});

const projectsPageSchema = z.object({
  title: z.string(),
  description: z.string(),
  projects: z.array(z.object({
    title: z.string(),
    description: z.string(),
    tags: z.array(z.string()),
    link: z.string(),
  })),
});

export const collections = {
  posts: defineCollection({
    type: 'content',
    schema: z.object({
      title: z.string(),
      date: z.date().optional(),
      modified: z.date().optional(),
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
  persona: defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
  research: defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
  about: defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
  'projects-page': defineCollection({
    type: 'content',
    schema: projectsPageSchema,
  }),
};