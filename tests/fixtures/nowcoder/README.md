# nowcoder fixtures

These are real responses captured from `nowcoder.com` on 2026-04-28 — kept here so
parsing tests can run offline and run-to-run deterministic.

| File | Source URL | Purpose |
|---|---|---|
| `sitemap_root.xml` | `/sitemap.xml` | Top-level sitemap index (points at sub-sitemaps) |
| `sitemap_nowpick.xml` | `/sitemap/nowpick/sitemap.xml` | nowpick sub-sitemap index |
| `sitemap_nowpick1.xml` | `/nowpick/sitemap1.xml` | 891 actual JD + enterprise URLs |
| `jd_446211.html` | `/jobs/detail/446211` | Sample SPA JD page; contains `window.__INITIAL_STATE__` JSON |

To refresh, re-run the curls in `docs/refresh_fixtures.sh` (TODO).
nowcoder allows crawling these pages per its `robots.txt`. We snapshot them only
to keep the test suite hermetic and fast — not for redistribution.
