/* OfferGuide Helper - background service worker.
 *
 * Two deliberately separate capabilities live here:
 * 1. Return an already prepared application package to the visible helper panel.
 * 2. Claim a server-created research request, open one inactive tab in this same
 *    signed-in Chrome profile, and return only rendered HTML/text/title/final URL.
 *
 * The bridge never returns cookies or storage, never clicks, scrolls, fills, or
 * submits, and closes only tabs it created itself.
 */

"use strict";

const OFFERGUIDE_ORIGIN = "http://localhost:8000";
const BRIDGE_ALARM = "offerguide-authenticated-browser-bridge";
const POLL_BURST_MS = 25000;
const POLL_INTERVAL_MS = 1500;
const PAGE_LOAD_TIMEOUT_MS = 40000;
const PAGE_STABILITY_TIMEOUT_MS = 12000;

let pollPromise = null;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.kind === "fetch_package_for_company") {
    const params = new URLSearchParams({company: msg.company});
    if (msg.job_id) params.set("job_id", String(msg.job_id));
    fetch(`${OFFERGUIDE_ORIGIN}/api/extension/package?${params.toString()}`)
      .then(async (response) => ({
        ok: response.ok,
        status: response.status,
        body: await response.json(),
      }))
      .then(({ok, status, body}) => {
        if (ok) {
          sendResponse({ok: true, package: body.package, job_id: body.job_id});
          return;
        }
        sendResponse({
          ok: false,
          status,
          error: body.error || `HTTP ${status}`,
          matches: body.matches || [],
        });
      })
      .catch((error) => sendResponse({ok: false, error: String(error), matches: []}));
    return true;
  }
  if (msg.kind === "wake_browser_bridge") {
    startPollingBurst();
    sendResponse({ok: true});
    return false;
  }
  return false;
});

chrome.runtime.onInstalled.addListener(() => {
  ensureBridgeAlarm();
  startPollingBurst();
});

chrome.runtime.onStartup.addListener(() => {
  ensureBridgeAlarm();
  startPollingBurst();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === BRIDGE_ALARM) startPollingBurst();
});

ensureBridgeAlarm();
startPollingBurst();

function ensureBridgeAlarm() {
  chrome.alarms.create(BRIDGE_ALARM, {periodInMinutes: 0.5});
}

function startPollingBurst() {
  if (pollPromise) return pollPromise;
  pollPromise = pollForRequests()
    .catch((error) => console.warn("OfferGuide browser bridge polling failed:", error))
    .finally(() => { pollPromise = null; });
  return pollPromise;
}

async function pollForRequests() {
  const startedAt = Date.now();
  let session;
  try {
    session = await connectBridge();
  } catch (error) {
    return;
  }

  while (Date.now() - startedAt < POLL_BURST_MS) {
    try {
      const request = await claimRequest(session);
      if (request) {
        await executeBrowserRequest(session, request);
        continue;
      }
    } catch (error) {
      console.warn("OfferGuide browser bridge request failed:", error);
      return;
    }
    await delay(POLL_INTERVAL_MS);
  }
}

async function connectBridge() {
  const response = await fetch(`${OFFERGUIDE_ORIGIN}/api/browser-bridge/config`, {
    method: "GET",
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`bridge config HTTP ${response.status}`);
  const config = await response.json();
  if (config.protocol_version !== 1 || !config.bridge_id || !config.access_token) {
    throw new Error("unsupported or incomplete browser bridge configuration");
  }

  const stored = await chrome.storage.local.get([
    "browserBridgeId",
    "browserBridgeClientId",
  ]);
  let clientId = stored.browserBridgeClientId;
  if (stored.browserBridgeId !== config.bridge_id || !validClientId(clientId)) {
    clientId = `og_${crypto.randomUUID().replaceAll("-", "")}`;
    await chrome.storage.local.set({
      browserBridgeId: config.bridge_id,
      browserBridgeClientId: clientId,
    });
  }
  const session = {config, clientId};
  const register = await bridgeFetch(session, "/api/browser-bridge/clients/register", {
    client_id: clientId,
    extension_version: chrome.runtime.getManifest().version,
  });
  if (!register.ok) throw new Error(`bridge registration HTTP ${register.status}`);
  return session;
}

async function claimRequest(session) {
  const response = await bridgeFetch(session, "/api/browser-bridge/requests/claim", {
    client_id: session.clientId,
  });
  if (response.status === 204) return null;
  if (!response.ok) throw new Error(`bridge claim HTTP ${response.status}`);
  return response.json();
}

async function executeBrowserRequest(session, request) {
  let openedTabId = null;
  try {
    assertPublicHttpShape(request.requested_url);
    const tab = await chrome.tabs.create({url: request.requested_url, active: false});
    if (typeof tab.id !== "number") throw new Error("browser did not create a tab");
    openedTabId = tab.id;

    await waitForTabComplete(openedTabId, request.deadline_at);
    const snapshot = await waitForStableSnapshot(openedTabId, request.deadline_at);
    assertPublicHttpShape(snapshot.finalUrl);

    if (snapshot.loginRequired) {
      await completeRequest(session, request, {
        status: "login_required",
        final_url: snapshot.finalUrl,
        title: snapshot.title,
        error_code: "login_required",
        error_text: "the signed-in browser reached a login page instead of readable source content",
      });
      return;
    }

    const htmlBytes = utf8Bytes(snapshot.html);
    const textBytes = utf8Bytes(snapshot.text);
    if (htmlBytes > session.config.max_rendered_html_bytes) {
      await completeRequest(session, request, {
        status: "rejected",
        final_url: snapshot.finalUrl,
        title: snapshot.title,
        error_code: "rendered_html_too_large",
        error_text: `rendered HTML is ${htmlBytes} bytes; limit is ${session.config.max_rendered_html_bytes}`,
      });
      return;
    }
    if (textBytes > session.config.max_rendered_text_bytes) {
      await completeRequest(session, request, {
        status: "rejected",
        final_url: snapshot.finalUrl,
        title: snapshot.title,
        error_code: "rendered_text_too_large",
        error_text: `rendered text is ${textBytes} bytes; limit is ${session.config.max_rendered_text_bytes}`,
      });
      return;
    }
    if (!snapshot.html || !snapshot.text.trim()) {
      await completeRequest(session, request, {
        status: "failed",
        final_url: snapshot.finalUrl,
        title: snapshot.title,
        error_code: "empty_rendered_page",
        error_text: "the rendered page contained no readable text",
      });
      return;
    }

    await completeRequest(session, request, {
      status: "succeeded",
      final_url: snapshot.finalUrl,
      title: snapshot.title,
      rendered_html: snapshot.html,
      rendered_text: snapshot.text,
    });
  } catch (error) {
    const code = error?.code || "browser_navigation_failed";
    await completeRequest(session, request, {
      status: code === "invalid_browser_url" ? "rejected" : "failed",
      error_code: code,
      error_text: String(error?.message || error).slice(0, 2000),
    }).catch((completionError) => {
      console.warn("OfferGuide could not report browser failure:", completionError);
    });
  } finally {
    if (openedTabId !== null) {
      await chrome.tabs.remove(openedTabId).catch(() => {});
    }
  }
}

async function waitForTabComplete(tabId, deadlineAtSeconds) {
  const requestRemaining = Math.max(1000, deadlineAtSeconds * 1000 - Date.now());
  const timeoutMs = Math.min(PAGE_LOAD_TIMEOUT_MS, requestRemaining);
  const current = await chrome.tabs.get(tabId);
  if (current.status === "complete") return;

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      reject(codedError("browser_load_timeout", "page did not finish loading before timeout"));
    }, timeoutMs);
    function listener(updatedTabId, changeInfo) {
      if (updatedTabId !== tabId) return;
      if (changeInfo.url) {
        try {
          assertPublicHttpShape(changeInfo.url);
        } catch (error) {
          clearTimeout(timer);
          chrome.tabs.onUpdated.removeListener(listener);
          chrome.tabs.remove(tabId).catch(() => {});
          reject(error);
          return;
        }
      }
      if (changeInfo.status === "complete") {
        clearTimeout(timer);
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    }
    chrome.tabs.onUpdated.addListener(listener);
  });
}

async function waitForStableSnapshot(tabId, deadlineAtSeconds) {
  const deadline = Math.min(
    Date.now() + PAGE_STABILITY_TIMEOUT_MS,
    deadlineAtSeconds * 1000 - 500,
  );
  let previousSignature = "";
  let stableCount = 0;
  while (Date.now() < deadline) {
    const snapshot = await extractSnapshot(tabId);
    const signature = [
      snapshot.readyState,
      snapshot.finalUrl,
      snapshot.html.length,
      snapshot.text.length,
    ].join("|");
    if (snapshot.readyState === "complete" && signature === previousSignature) {
      stableCount += 1;
      if (stableCount >= 2) return snapshot;
    } else {
      stableCount = 0;
      previousSignature = signature;
    }
    await delay(650);
  }
  throw codedError("render_timeout", "rendered page did not reach a stable complete state");
}

async function extractSnapshot(tabId) {
  const results = await chrome.scripting.executeScript({
    target: {tabId},
    world: "ISOLATED",
    func: () => {
      const helper = document.getElementById("offerguide-helper-panel");
      const helperText = helper?.innerText || "";
      let text = document.body?.innerText || "";
      if (helperText && text.includes(helperText)) text = text.replace(helperText, "");

      const clone = document.documentElement.cloneNode(true);
      clone.querySelector("#offerguide-helper-panel")?.remove();
      const passwordVisible = Array.from(document.querySelectorAll('input[type="password"]'))
        .some((element) => element.getClientRects().length > 0);
      const loginPath = /(^|\/)(login|signin|sign-in|passport|auth)(\/|$)/i
        .test(window.location.pathname);
      const loginLanguage = /(请先?登录|登录后|sign\s*in|log\s*in)/i
        .test(text.slice(0, 5000));
      const explicitLoginWall = /(请先?登录|登录后(?:查看|继续|访问)|sign\s*in\s+to\s+(?:continue|view|access))/i
        .test(text.slice(0, 5000));
      return {
        readyState: document.readyState,
        finalUrl: window.location.href,
        title: document.title || "",
        html: clone.outerHTML,
        text,
        loginRequired: passwordVisible || explicitLoginWall || (loginPath && loginLanguage),
      };
    },
  });
  const snapshot = results?.[0]?.result;
  if (!snapshot) throw codedError("page_not_scriptable", "browser could not read the rendered page");
  return snapshot;
}

async function completeRequest(session, request, result) {
  const response = await bridgeFetch(
    session,
    `/api/browser-bridge/requests/${encodeURIComponent(request.request_id)}/complete`,
    {
      client_id: session.clientId,
      lease_token: request.lease_token,
      ...result,
    },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`bridge completion HTTP ${response.status}: ${detail.slice(0, 500)}`);
  }
}

function bridgeFetch(session, path, body) {
  return fetch(`${OFFERGUIDE_ORIGIN}${path}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${session.config.access_token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });
}

function assertPublicHttpShape(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (error) {
    throw codedError("invalid_browser_url", "browser request URL is invalid");
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    throw codedError("invalid_browser_url", "browser request must be a credential-free HTTP(S) URL");
  }
  const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (
    host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local") ||
    /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host) || /^169\.254\./.test(host) ||
    /^0\./.test(host) || host === "::1" || /^f[cd][0-9a-f]*:/i.test(host) ||
    /^fe[89ab][0-9a-f]*:/i.test(host)
  ) {
    throw codedError("invalid_browser_url", "local or private browser targets are blocked");
  }
}

function codedError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function validClientId(value) {
  return typeof value === "string" && /^[A-Za-z0-9_-]{16,128}$/.test(value);
}

function utf8Bytes(value) {
  return new TextEncoder().encode(value).byteLength;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
