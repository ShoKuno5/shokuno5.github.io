CREATE TABLE IF NOT EXISTS analytics_events (
  day TEXT NOT NULL,
  event TEXT NOT NULL,
  path TEXT NOT NULL,
  variant TEXT NOT NULL DEFAULT '',
  count INTEGER NOT NULL DEFAULT 0,
  last_seen TEXT NOT NULL,
  PRIMARY KEY (day, event, path, variant)
);

CREATE INDEX IF NOT EXISTS idx_analytics_event_day ON analytics_events(event, day);
CREATE INDEX IF NOT EXISTS idx_analytics_path_day ON analytics_events(path, day);
