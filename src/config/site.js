export const SITE = {
  name: 'Sho Kuno',
  tagline: 'Building up: research, projects, fragments, notes.',
  posts: {
    excludeTags: ['draft', 'private'],
  },
  analytics: {
    endpoint: '',
  },
  contact: {
    email: 'kunosho1225@g.ecc.u-tokyo.ac.jp',
    github: 'https://github.com/ShoKuno5',
    linkedin: 'https://www.linkedin.com/in/sho-kuno-828a0133a/',
    scholar: 'https://scholar.google.com/citations?user=Vhz0T3kAAAAJ&hl=ja',
    x: 'https://x.com/ShoKunoR',
    cv: '/cv.pdf',
  },
};

export const NAV_LINKS = [
  { label: 'Home', href: '/' },
  { label: 'Research', href: '/research/' },
  { label: 'Topics', href: '/topics/' },
  { label: 'Series', href: '/series/' },
  { label: 'Tags', href: '/tags/' },
];

/** @type {{ key: string; label: string; href: string; locations?: string[] }[]} */
export const CONTACT_LINKS = [];

if (SITE.contact?.email) {
  CONTACT_LINKS.push({
    key: 'email',
    label: 'Email',
    href: `mailto:${SITE.contact.email}`,
    locations: ['nav', 'post'],
  });
}

if (SITE.contact?.github) {
  CONTACT_LINKS.push({
    key: 'github',
    label: 'GitHub',
    href: SITE.contact.github,
    locations: ['nav', 'post'],
  });
}

if (SITE.contact?.linkedin) {
  CONTACT_LINKS.push({
    key: 'linkedin',
    label: 'LinkedIn',
    href: SITE.contact.linkedin,
    locations: ['nav', 'post'],
  });
}

if (SITE.contact?.scholar) {
  CONTACT_LINKS.push({
    key: 'scholar',
    label: 'Google Scholar',
    href: SITE.contact.scholar,
    locations: ['nav', 'post'],
  });
}

if (SITE.contact?.x) {
  CONTACT_LINKS.push({
    key: 'x',
    label: 'X',
    href: SITE.contact.x,
    locations: ['nav', 'post'],
  });
}

if (SITE.contact?.cv) {
  CONTACT_LINKS.push({
    key: 'cv',
    label: 'CV',
    href: SITE.contact.cv,
    locations: ['nav', 'post'],
  });
}
