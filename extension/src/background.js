/* OfferGuide Helper — background service worker (W13.8)
 *
 * Security stance:
 * - Talks ONLY to localhost:8000 (your own OfferGuide instance) and the
 *   declared host_permissions sites
 * - Never sends ANY platform DOM data to OfferGuide; we only fetch ALREADY-
 *   prepared application packages from local DB
 * - Never automates clicks / submits / navigation on platform pages
 * - Per-tab rate limiter keeps copy operations under a human-plausible
 *   threshold (≤ 5/min). Boss直聘's anti-bot doesn't care about clipboard
 *   reads, but rate is still our ethical floor.
 */

const RATE_LIMIT_PER_MIN = 5;
const RATE_LIMIT_PER_HOUR = 30;

// In-memory rate counter (cleared on service-worker restart, which is fine —
// chrome lifecycle ≈ a few minutes idle, well above our 1-min window)
const recentCopies = []; // [{ts: ms, tabId}]

function _purgeOldCopies(now) {
  const oneHourAgo = now - 60 * 60 * 1000;
  let i = 0;
  while (i < recentCopies.length && recentCopies[i].ts < oneHourAgo) i++;
  if (i > 0) recentCopies.splice(0, i);
}

function checkRateLimit(tabId) {
  const now = Date.now();
  _purgeOldCopies(now);
  const oneMinAgo = now - 60_000;
  const lastMinute = recentCopies.filter(c => c.ts > oneMinAgo);
  if (lastMinute.length >= RATE_LIMIT_PER_MIN) {
    return {ok: false, reason: `rate_limit_min: ${lastMinute.length}/min`};
  }
  if (recentCopies.length >= RATE_LIMIT_PER_HOUR) {
    return {ok: false, reason: `rate_limit_hour: ${recentCopies.length}/hr`};
  }
  return {ok: true};
}

function recordCopy(tabId) {
  recentCopies.push({ts: Date.now(), tabId});
}

// Message router: content script asks background to fetch from localhost
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.kind === "fetch_package_for_company") {
    // Look up an apply package by company name (the content script has
    // sniffed the page title / URL to know which company we're on)
    fetch(`http://localhost:8000/api/extension/package?company=${encodeURIComponent(msg.company)}`)
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(data => sendResponse({ok: true, package: data.package, job_id: data.job_id}))
      .catch(err => sendResponse({ok: false, error: String(err)}));
    return true; // keep channel open for async response
  }
  if (msg.kind === "check_rate") {
    sendResponse(checkRateLimit(sender.tab?.id || 0));
    return false;
  }
  if (msg.kind === "record_copy") {
    recordCopy(sender.tab?.id || 0);
    sendResponse({ok: true, totalThisHour: recentCopies.length});
    return false;
  }
  if (msg.kind === "get_recent_count") {
    _purgeOldCopies(Date.now());
    const oneMinAgo = Date.now() - 60_000;
    sendResponse({
      lastMinute: recentCopies.filter(c => c.ts > oneMinAgo).length,
      lastHour: recentCopies.length,
    });
    return false;
  }
  return false;
});
