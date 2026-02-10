const ALLOWED_EVENTS = new Set([
  'page_view',
  'scroll_50',
  'scroll_90',
  'citation_open',
  'comments_load',
  'search_use',
  'related_post_click',
  'citation_graph_click',
]);

const json = (payload, status = 200, headers = {}) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...headers,
    },
  });

const normalizePath = (value = '/') => {
  if (typeof value !== 'string' || !value.trim()) return '/';
  try {
    const url = new URL(value, 'https://local');
    const path = url.pathname || '/';
    return path.startsWith('/') ? path : `/${path}`;
  } catch {
    const trimmed = value.split('?')[0].split('#')[0].trim();
    if (!trimmed) return '/';
    return trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  }
};

const normalizeVariant = (meta = {}) => {
  if (!meta || typeof meta !== 'object') return '';
  const entries = Object.entries(meta)
    .filter(([key, value]) => Boolean(key) && value != null)
    .slice(0, 4)
    .map(([key, value]) => `${String(key).slice(0, 32)}=${String(value).slice(0, 64)}`);
  return entries.join('|').slice(0, 200);
};

const corsHeaders = (origin, env) => {
  const allowedOrigin = env.ALLOWED_ORIGIN || '*';
  const value = allowedOrigin === '*' ? '*' : origin || allowedOrigin;
  return {
    'access-control-allow-origin': value,
    'access-control-allow-methods': 'POST, OPTIONS',
    'access-control-allow-headers': 'content-type',
    'access-control-max-age': '86400',
    vary: 'origin',
  };
};

async function ingestEvent(request, env) {
  const origin = request.headers.get('origin') || '';
  const cors = corsHeaders(origin, env);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: 'invalid_json' }, 400, cors);
  }

  const event = String(body?.event || '').trim();
  if (!ALLOWED_EVENTS.has(event)) {
    return json({ ok: false, error: 'event_not_allowed' }, 400, cors);
  }

  const path = normalizePath(body?.path || '/');
  const variant = normalizeVariant(body?.meta);
  const now = new Date();
  const day = now.toISOString().slice(0, 10);
  const seenAt = now.toISOString();

  await env.DB.prepare(
    `INSERT INTO analytics_events (day, event, path, variant, count, last_seen)
      VALUES (?1, ?2, ?3, ?4, 1, ?5)
      ON CONFLICT(day, event, path, variant)
      DO UPDATE SET count = count + 1, last_seen = excluded.last_seen`
  )
    .bind(day, event, path, variant, seenAt)
    .run();

  return json({ ok: true }, 202, cors);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('origin') || '';
    const cors = corsHeaders(origin, env);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    const url = new URL(request.url);

    if (request.method === 'POST' && url.pathname === '/v1/event') {
      return ingestEvent(request, env);
    }

    if (request.method === 'GET' && url.pathname === '/health') {
      return json({ ok: true }, 200, cors);
    }

    return json({ ok: false, error: 'not_found' }, 404, cors);
  },
};
