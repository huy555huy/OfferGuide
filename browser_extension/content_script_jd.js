/**
 * OfferGuide BOSS JD content script (W15.20).
 *
 * 当用户在 BOSS 直聘 JD 详情页 (zhipin.com/job_detail/...) 浏览时,
 * 这个脚本自动抓 JD → 调 OfferGuide /api/extension/score_inline → 在
 * 页面右上角注入一个浮动 OfferGuide 评分卡. 用户能立刻看到 score +
 * 关键 gap, 不用切回 OfferGuide 页面.
 *
 * 卡片有"📋 写开场白"按钮: 点 → 调 /api/extension/greeting → 把 200 字
 * 开场白写到剪贴板, 用户粘到 BOSS 沟通框 (审核 + 改抬头后再发).
 *
 * 设计原则:
 * - 不自动发送任何东西 — 只评分 + 写文案到剪贴板
 * - Shadow DOM 隔离 — 不被 BOSS CSS 污染, 不污染 BOSS 页面
 * - Fail silent — 后端没起 / 没简历 / 出错 都只显示提示, 不打扰浏览
 * - 防重复触发 — 同一 URL 只跑一次 (用 dataset flag)
 */

/* global chrome */

const API_BASE = "http://localhost:8000";
const HOST_ATTR = "data-offerguide-injected";

// ── 提取 JD (复用 popup.js 的同名逻辑, 但独立 self-contained) ──────────

function extractBossJDInline() {
  function text(sel) {
    const el = document.querySelector(sel);
    return el ? el.textContent.trim() : "";
  }
  function trySelectors(selectors) {
    for (const sel of selectors) {
      const t = text(sel);
      if (t) return t;
    }
    return "";
  }

  const title = trySelectors([
    ".job-banner .name h1",
    ".job-title .name",
    "[class*='job-name'] h1",
    "[class*='job-name']",
  ]) || document.title.split("-")[0].trim();

  const company = trySelectors([
    ".company-info .name",
    ".company-info a",
    "[class*='company-name']",
  ]);

  const salary = trySelectors([
    ".job-banner .salary",
    ".salary",
    "[class*='salary']",
  ]);

  const location = trySelectors([
    ".job-primary .info-primary p",
    ".location-address .job-location",
  ]).split("\n")[0];

  let description = "";
  for (const sel of [
    ".job-detail .job-sec-text",
    ".job-sec-text",
    "[class*='job-detail-section']",
    ".text.fold-text",
    ".text",
  ]) {
    const el = document.querySelector(sel);
    if (el && el.innerText.trim().length > 50) {
      description = el.innerText.trim();
      break;
    }
  }

  const tags = Array.from(
    document.querySelectorAll(".job-tags .tag-item, .tag-list .tag, [class*='job-tag']")
  ).map(el => el.textContent.trim()).filter(Boolean);

  return {
    url: window.location.href,
    title, company, salary, location, description, tags,
  };
}

// ── UI 注入 ──────────────────────────────────────────────────────────

function buildHostShell() {
  const host = document.createElement("div");
  host.id = "offerguide-host";
  host.setAttribute(HOST_ATTR, "1");
  host.style.cssText =
    "position:fixed;top:80px;right:24px;z-index:2147483640;width:320px;";
  const root = host.attachShadow({ mode: "open" });

  const styles = document.createElement("style");
  styles.textContent = `
    .card{background:#fff;border:1px solid #e0d8c8;border-radius:10px;
      box-shadow:0 6px 20px rgba(0,0,0,.10);font-family:-apple-system,
      "PingFang SC",sans-serif;font-size:13px;color:#222;overflow:hidden}
    .hdr{padding:10px 14px;display:flex;align-items:center;
      justify-content:space-between;border-bottom:1px solid #f0e8d8;
      background:linear-gradient(#fff,#fbf6ed)}
    .hdr .brand{font-weight:600;font-size:13px;color:#cc785c}
    .hdr .closebtn{cursor:pointer;color:#999;border:none;background:none;
      font-size:16px;line-height:1;padding:0 4px}
    .body{padding:12px 14px}
    .row{margin-bottom:8px}
    .score{font-size:32px;font-weight:700;line-height:1}
    .score.green{color:#137333}.score.yellow{color:#b06000}
    .score.red{color:#c5221f}.score.gray{color:#888}
    .verdict{display:inline-block;margin-left:8px;padding:2px 8px;
      border-radius:999px;font-size:11px;font-weight:500}
    .verdict.green{background:#e6f4ea;color:#137333}
    .verdict.yellow{background:#fef7e0;color:#b06000}
    .verdict.red{background:#fce8e6;color:#c5221f}
    .verdict.gray{background:#eee;color:#666}
    .label{font-size:11px;color:#888;text-transform:uppercase;
      letter-spacing:.04em;margin-bottom:3px}
    .gap{padding:3px 0;border-top:1px dotted #eee;color:#444;font-size:12px}
    .gap:first-of-type{border-top:none}
    .actions{margin-top:10px;display:flex;gap:6px}
    button.primary{flex:1;padding:8px;border:none;border-radius:5px;
      cursor:pointer;background:#cc785c;color:#fff;font-size:12px;
      font-weight:500}
    button.primary:hover{background:#b8654a}
    button.primary:disabled{background:#ccc;cursor:not-allowed}
    button.secondary{padding:8px 10px;border:1px solid #ddd;border-radius:5px;
      cursor:pointer;background:#fafafa;color:#444;font-size:12px}
    .status{margin-top:8px;padding:6px 8px;border-radius:4px;font-size:11px;
      line-height:1.45}
    .status.ok{background:#e6f4ea;color:#137333}
    .status.err{background:#fce8e6;color:#c5221f}
    .status.info{background:#e8f4fd;color:#1a73e8}
    .greet-box{margin-top:8px;padding:8px;background:#f6f3ee;border-radius:5px;
      font-size:12px;line-height:1.5;white-space:pre-wrap;color:#333;
      max-height:200px;overflow-y:auto}
    .probe-block{margin-top:12px;padding-top:10px;border-top:1px dashed #e0d8c8}
    .muted{color:#888;font-size:11px}
    a{color:#1a73e8;text-decoration:none}a:hover{text-decoration:underline}
  `;
  root.appendChild(styles);

  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <div class="hdr">
      <div class="brand">🎯 OfferGuide</div>
      <button class="closebtn" id="og-close" title="关闭">✕</button>
    </div>
    <div class="body">
      <div id="og-content">
        <div class="status info">正在评估这个岗位…</div>
      </div>
    </div>
  `;
  root.appendChild(card);
  return { host, root };
}

function renderScore(root, data) {
  const content = root.getElementById("og-content");
  const scoreVal = data.score == null ? "—" : Math.round(data.score);
  const color = data.color || "gray";
  const verdict = data.verdict || "";
  const gaps = (data.top_gaps || []).slice(0, 3);

  let gapsHtml = "";
  if (gaps.length > 0) {
    gapsHtml = `
      <div class="row">
        <div class="label">关键 gap</div>
        ${gaps.map(g => `<div class="gap">• ${escapeHtml(g)}</div>`).join("")}
      </div>
    `;
  }

  content.innerHTML = `
    <div class="row">
      <span class="score ${color}">${scoreVal}</span>
      ${verdict ? `<span class="verdict ${color}">${escapeHtml(verdict)}</span>` : ""}
      <div class="muted" style="margin-top:4px">
        ${data.duration_ms ? Math.round(data.duration_ms / 100) / 10 + "s" : ""}
        ${data.cost_usd ? " · $" + data.cost_usd.toFixed(4) : ""}
      </div>
    </div>
    ${gapsHtml}
    <div class="actions">
      <button class="primary" id="og-greet">📋 写开场白</button>
      <button class="secondary" id="og-open">待点列表</button>
    </div>
    <div id="og-greet-status"></div>
    <div id="og-greet-box"></div>
    <div class="probe-block">
      <div class="muted" style="font-size:11px;line-height:1.5">
        🔧 帮 OfferGuide 升级: 你打开 BOSS 沟通框后, 点下面按钮抓一次 DOM →
        我们后续就能自动填开场白到沟通框 (现在还是手粘).
      </div>
      <button class="secondary" id="og-probe" style="margin-top:6px;width:100%">
        🔍 抓沟通框 DOM
      </button>
      <div id="og-probe-status"></div>
    </div>
  `;

  // Wire actions
  root.getElementById("og-greet").addEventListener("click", () => {
    runGreeting(root, data.job_id);
  });
  root.getElementById("og-open").addEventListener("click", () => {
    window.open(API_BASE + "/recommended", "_blank");
  });
  root.getElementById("og-probe").addEventListener("click", () => {
    runProbe(root);
  });
}

async function runProbe(root) {
  const statusEl = root.getElementById("og-probe-status");
  const btn = root.getElementById("og-probe");
  btn.disabled = true;
  statusEl.innerHTML = `<div class="status info">正在找沟通框…</div>`;

  // Try a wide selector net for the chat input area (BOSS uses several)
  const chatSelectors = [
    ".chat-record", ".chat-im-wrap", ".dialog-im",
    "[class*='chat-conversation']", "[class*='chat-input']",
    ".chat-textarea", "textarea[placeholder*='请输入']",
    ".chat-send-area", ".message-controller",
  ];

  let snippet = null;
  let kind = "chat_box";
  for (const sel of chatSelectors) {
    const el = document.querySelector(sel);
    if (el) {
      // Walk up to find a meaningful container (not just the input alone)
      let host = el.closest(".chat-im-wrap, .dialog-wrap, [class*='chat']") || el.parentElement || el;
      snippet = host.outerHTML.slice(0, 100_000);
      break;
    }
  }
  if (!snippet) {
    // Fallback: check if there's any visible chat-like element on page
    const possibleChat = document.querySelector("[class*='chat'],[class*='dialog'],[class*='im-']");
    if (possibleChat) {
      snippet = possibleChat.outerHTML.slice(0, 100_000);
      kind = "chat_unknown";
    }
  }
  if (!snippet) {
    statusEl.innerHTML =
      `<div class="status err">没找到沟通框 — 先在 BOSS 点"立即沟通"打开聊天, 再点这里</div>`;
    btn.disabled = false;
    return;
  }

  try {
    const resp = await fetch(API_BASE + "/api/extension/probe_dom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: window.location.href,
        snippet_kind: kind,
        outer_html: snippet,
        captured_at: new Date().toISOString(),
      }),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      statusEl.innerHTML =
        `<div class="status err">上传失败 (${resp.status}): ${escapeHtml(txt.slice(0, 80))}</div>`;
      btn.disabled = false;
      return;
    }
    const data = await resp.json();
    statusEl.innerHTML =
      `<div class="status ok">✓ 已抓 ${Math.round(snippet.length/1024)}KB → ${escapeHtml(data.saved_as || "")}</div>`;
    btn.textContent = "🔄 再抓一次";
    btn.disabled = false;
  } catch (err) {
    statusEl.innerHTML =
      `<div class="status err">连接失败: ${escapeHtml(err.message)}</div>`;
    btn.disabled = false;
  }
}

function renderError(root, msg, hint) {
  const content = root.getElementById("og-content");
  content.innerHTML = `
    <div class="status err">${escapeHtml(msg)}</div>
    ${hint ? `<div class="muted" style="margin-top:6px">${escapeHtml(hint)}</div>` : ""}
  `;
}

async function runGreeting(root, jobId) {
  const statusEl = root.getElementById("og-greet-status");
  const boxEl = root.getElementById("og-greet-box");
  const btn = root.getElementById("og-greet");
  btn.disabled = true;
  statusEl.innerHTML = `<div class="status info">正在写开场白 (5-10s)…</div>`;
  boxEl.innerHTML = "";

  try {
    const resp = await fetch(API_BASE + "/api/extension/greeting", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      statusEl.innerHTML =
        `<div class="status err">写开场白失败 (${resp.status}): ${escapeHtml(txt.slice(0, 100))}</div>`;
      btn.disabled = false;
      return;
    }
    const data = await resp.json();
    const greeting = data.greeting || "";

    boxEl.innerHTML = `
      <div class="greet-box">${escapeHtml(greeting)}</div>
    `;

    // Try writing to clipboard
    try {
      await navigator.clipboard.writeText(greeting);
      statusEl.innerHTML =
        `<div class="status ok">✓ 已复制 (${greeting.length} 字) — Cmd+V 到 BOSS 沟通框, 改抬头再发</div>`;
    } catch (clipErr) {
      statusEl.innerHTML =
        `<div class="status info">已生成 (${greeting.length} 字) — 剪贴板权限被拦, 手动选中复制</div>`;
    }
    btn.disabled = false;
    btn.textContent = "🔄 重新生成";
  } catch (err) {
    statusEl.innerHTML =
      `<div class="status err">调用失败: ${escapeHtml(err.message)}</div>`;
    btn.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ── main ────────────────────────────────────────────────────────────

function renderProbeOnly(root) {
  const content = root.getElementById("og-content");
  content.innerHTML = `
    <div class="row">
      <div class="muted" style="font-size:12px;line-height:1.5">
        💬 检测到沟通页. OfferGuide 还不会自动填开场白进去 — DOM 选择器
        没确认. 你帮我抓一次, 下版本就能自动填了.
      </div>
    </div>
    <div class="probe-block" style="margin-top:8px;padding-top:0;border-top:none">
      <button class="primary" id="og-probe" style="width:100%">🔍 抓沟通框 DOM 给 OfferGuide</button>
      <div id="og-probe-status"></div>
    </div>
  `;
  root.getElementById("og-probe").addEventListener("click", () => {
    runProbe(root);
  });
}

async function main() {
  // 防重复注入 (BOSS SPA 切页时这脚本可能跑多次)
  if (document.querySelector(`[${HOST_ATTR}]`)) return;

  const url = window.location.href;
  const isJD = /\/job_detail\//.test(url);
  const isChat = /\/web\/chat\//.test(url);
  if (!isJD && !isChat) return;

  // 等 1.2s 让 BOSS DOM 渲染完
  await new Promise(r => setTimeout(r, 1200));

  // 沟通页: 只装 probe 浮窗, 不评分
  if (isChat) {
    const { host, root } = buildHostShell();
    document.body.appendChild(host);
    root.getElementById("og-close").addEventListener("click", () => host.remove());
    renderProbeOnly(root);
    return;
  }

  const jd = extractBossJDInline();
  if (!jd.description || jd.description.length < 100) {
    // JD 没加载完, 别注入
    return;
  }

  const { host, root } = buildHostShell();
  document.body.appendChild(host);
  root.getElementById("og-close").addEventListener("click", () => {
    host.remove();
  });

  // 调 score_inline
  try {
    const resp = await fetch(API_BASE + "/api/extension/score_inline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jd),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      let msg = "评估失败";
      try {
        const errJson = JSON.parse(txt);
        msg = errJson.error || msg;
      } catch (_) {
        msg = txt.slice(0, 120) || msg;
      }
      let hint = "";
      if (resp.status === 400 && msg.includes("简历")) {
        hint = "去 OfferGuide 上传简历 → 重刷本页";
      } else if (resp.status === 400 && msg.includes("LLM key")) {
        hint = "在 .env 配 DEEPSEEK_API_KEY";
      }
      renderError(root, msg, hint);
      return;
    }
    const data = await resp.json();
    renderScore(root, data);
  } catch (err) {
    renderError(
      root,
      "连不上 OfferGuide 后端",
      "确认 python -m offerguide.ui.web 起着 (localhost:8000)"
    );
  }
}

// BOSS 是 SPA, popstate 可能切到不同 JD 也不刷新整页
let lastUrl = window.location.href;
new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    // 旧 host 移除, 重新注入
    const old = document.querySelector(`[${HOST_ATTR}]`);
    if (old) old.remove();
    setTimeout(main, 800);
  }
}).observe(document.body, { childList: true, subtree: true });

main();
