import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import type { CollectionEntry } from 'astro:content';

type GitDates = {
  published: Date | null;
  updated: Date | null;
};

const gitDateCache = new Map<string, GitDates>();
const repoRoot = process.cwd();
const postsDir = path.join(repoRoot, 'src', 'content', 'posts');

const parseGitDate = (value: string): Date | null => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const getFileMtime = (filePath: string): Date | null => {
  try {
    return fs.statSync(filePath).mtime;
  } catch {
    return null;
  }
};

const getGitDates = (filePath: string): GitDates => {
  if (gitDateCache.has(filePath)) {
    return gitDateCache.get(filePath) ?? { published: null, updated: null };
  }

  let published: Date | null = null;
  let updated: Date | null = null;

  try {
    const output = execFileSync('git', ['log', '--follow', '--format=%cI', '--', filePath], {
      cwd: repoRoot,
    })
      .toString()
      .trim();

    if (output) {
      const lines = output
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);

      if (lines.length > 0) {
        updated = parseGitDate(lines[0]);
        published = parseGitDate(lines[lines.length - 1]);
      }
    }
  } catch {
    published = null;
    updated = null;
  }

  const resolved = { published, updated };
  gitDateCache.set(filePath, resolved);
  return resolved;
};

export const getPostDates = (
  entry: CollectionEntry<'posts'>
): { published: Date; updated: Date } => {
  const filePath = path.join(postsDir, entry.id);
  const { published: gitPublished, updated: gitUpdated } = getGitDates(filePath);
  const fallback = entry.data.pubDate ?? getFileMtime(filePath) ?? new Date(0);

  const published = gitPublished ?? fallback;
  const updated = gitUpdated ?? published;

  return { published, updated };
};
