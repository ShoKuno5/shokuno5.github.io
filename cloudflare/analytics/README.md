# Privacy-First Analytics (Cloudflare Worker)

This worker ingests minimal event telemetry and stores **daily aggregated counts** in D1.
It does not use cookies, user IDs, or fingerprinting.

## 1) Create a D1 database

```bash
wrangler d1 create myfolio_analytics
```

## 2) Apply schema

```bash
wrangler d1 execute myfolio_analytics --file ./cloudflare/analytics/schema.sql
```

## 3) Configure `wrangler.toml`

```toml
name = "myfolio-analytics"
main = "cloudflare/analytics/worker.js"
compatibility_date = "2026-02-10"

[[d1_databases]]
binding = "DB"
database_name = "myfolio_analytics"
database_id = "<YOUR_D1_DATABASE_ID>"

[vars]
ALLOWED_ORIGIN = "https://shokuno5.github.io"
```

## 4) Deploy

```bash
wrangler deploy
```

Your ingest endpoint will be:

```text
https://<your-worker-subdomain>/v1/event
```

## 5) Connect site

Set `SITE.analytics.endpoint` in `src/config/site.js` to your worker endpoint.

## Events currently emitted

- `page_view`
- `scroll_50`
- `scroll_90`
- `citation_open`
- `comments_load`
- `related_post_click`
- `citation_graph_click`

## Notes

- Keep this endpoint first-party and same-origin if possible.
- Add a public `/privacy` page that describes event collection and no-cookie policy.
