const CITE_PATTERN = /\\cite\{([^}]+)\}/g;

export function extractCitationKeys(source: string = ''): string[] {
  if (!source) return [];

  const keys: string[] = [];
  const seen = new Set<string>();
  let match: RegExpExecArray | null;

  CITE_PATTERN.lastIndex = 0;

  while ((match = CITE_PATTERN.exec(source)) !== null) {
    const raw = match[1] || '';
    const parts = raw
      .split(',')
      .map((part) => part.trim())
      .filter(Boolean);

    for (const key of parts) {
      if (seen.has(key)) continue;
      seen.add(key);
      keys.push(key);
    }
  }

  return keys;
}
