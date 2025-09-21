// Single source of truth for site-wide labels and routes

export const LOCALES = ['en'];
export const DEFAULT_LOCALE = 'en';

export const SITE = {
  name: 'face',
  contact: {
    emailHref: 'mailto:kunosho1225@g.ecc.u-tokyo.ac.jp',
  },
};

const route = (base, locale) => (locale === 'ja' ? `/ja${base}` : base);

// Section metadata keyed by logical section id
export const SECTIONS = {
  face: {
    id: 'face',
    labels: { en: 'Face', ja: 'Face' },
    // Home page
    path: (locale = DEFAULT_LOCALE) => (locale === 'ja' ? '/ja/' : '/'),
  },
  profile: {
    id: 'profile',
    collection: 'profile',
    slug: 'profile',
    labels: { en: 'Profile', ja: 'プロフィール' },
    path: (locale = DEFAULT_LOCALE) => route('/profile/', locale),
  },
  research: {
    id: 'research',
    collection: 'research',
    slug: 'research',
    labels: { en: 'Salients', ja: '研究' },
    path: (locale = DEFAULT_LOCALE) => route('/research/', locale),
  },
  projects: {
    id: 'projects',
    labels: { en: 'Projects', ja: 'プロジェクト' },
    path: (locale = DEFAULT_LOCALE) => route('/projects/', locale),
  },
  posts: {
    id: 'posts',
    labels: { en: 'Posts', ja: '投稿' },
    allLabel: { en: 'All Posts', ja: 'すべての投稿' },
    tagsLabel: { en: 'All Tags', ja: 'すべてのタグ' },
    path: (locale = DEFAULT_LOCALE) => route('/posts/', locale),
  },
  contacts: {
    id: 'contacts',
    labels: { en: 'Contacts', ja: '連絡先' },
    path: (locale = DEFAULT_LOCALE) => route('/contacts/', locale),
  },
  about: {
    id: 'about',
    labels: { en: 'About', ja: '概要' },
    path: (locale = DEFAULT_LOCALE) => route('/about/', locale),
  },
};

// Navigation items for top-level sections
export function getNav(locale = DEFAULT_LOCALE) {
  return [
    { label: SECTIONS.face.labels[locale], href: SECTIONS.face.path(locale) },
    { label: SECTIONS.profile.labels[locale], href: SECTIONS.profile.path(locale) },
    { label: SECTIONS.posts.labels[locale], href: SECTIONS.posts.path(locale) },
    { label: SECTIONS.research.labels[locale], href: SECTIONS.research.path(locale) },
    { label: SECTIONS.contacts.labels[locale], href: SECTIONS.contacts.path(locale) },
  ];
}

// Static paths used in the sitemap
export const STATIC_PATHS = [
  '/',
  SECTIONS.contacts.path('en'),
  SECTIONS.profile.path('en'),
  SECTIONS.about.path('en'),
  SECTIONS.projects.path('en'),
  '/posts/all/',
  '/posts/tags/',
  SECTIONS.research.path('en'),
];
