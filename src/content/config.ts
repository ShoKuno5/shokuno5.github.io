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
      pubDate: z.date(),
      date: z.date().optional(),
      modified: z.date().optional(),
      description: z.string().optional(),
      author: z.string().optional(),
      tags: z.array(z.string()).optional(),
      summary: z.string().optional(),
      hidden: z.boolean().optional(),
      layout: z.string().optional(),
      type: z.string().optional(),
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
  profile: defineCollection({
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
  face: defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
  contacts: defineCollection({
    type: 'content',
    schema: staticPageSchema,
  }),
};
