/**
 * OfferGuide Boss 助手 — popup script (W15.18 multi-mode).
 *
 * 支持 2 个 mode:
 *
 * 1. **JD 详情模式** (zhipin.com/job_detail/...)
 *    单个岗位 → 提取标题/公司/薪资/JD → 1 条入库
 *    现状: v0.1 已支持
 *
 * 2. **推荐列表模式** (zhipin.com/web/geek/recommend, /jobs.html, /web/geek/jobs)
 *    用户 BOSS 推荐池 → 提取列表里所有岗位 → bulk 入库
 *    新增 v0.2 — 用户在 BOSS 自己刷, 一键 sync 整页到 OfferGuide.
 *    Agent 拿到后自动 score / 排序 / 主动通知.
 *
 * 设计原则: 用户必须**手动点击扩展**才提取. 没有 background script /
 * content_script auto-fire — 用户控制一切.
 */

/* global chrome */

const API_BASE = "http://localhost:8000";

// URL → mode 映射
const URL_PATTERNS = {
  jd_detail: /^https?:\/\/(www\.)?zhipin\.com\/job_detail\//,
  // 推荐 / 搜索 / 列表页常见 URL 模式
  recommend_list: /^https?:\/\/(www\.)?zhipin\.com\/(web\/geek\/(recommend|jobs)|jobs\.html|c\d+)/,
};

// ── DOM refs ───────────────────────────────────────────────────────

const statusEl = document.getElementById("status");
const modePill = document.getElementById("mode-pill");
const fieldsDetail = document.getElementById("fields-detail");
const fieldsList = document.getElementById("fields-list");
const listSummary = document.getElementById("list-summary");
const listItems = document.getElementById("list-items");
const btnSend = document.getElementById("btn-send");
const btnRefresh = document.getElementById("btn-refresh");
const fTitle = document.getElementById("f-title");
const fCompany = document.getElementById("f-company");
const fSalary = document.getElementById("f-salary");
const fLocation = document.getElementById("f-location");
const fDesc = document.getElementById("f-desc");
const fTags = document.getElementById("f-tags");

let currentMode = null;  // 'jd_detail' | 'recommend_list'
let extractedData = null;  // mode-specific shape

function setStatus(cls, msg) {
  statusEl.className = "status " + cls;
  statusEl.textContent = msg;
}

function setMode(mode, label) {
  currentMode = mode;
  if (label) {
    modePill.textContent = label;
    modePill.classList.remove("hidden");
  } else {
    modePill.classList.add("hidden");
  }
  fieldsDetail.classList.toggle("hidden", mode !== "jd_detail");
  fieldsList.classList.toggle("hidden", mode !== "recommend_list");
}

// ═══════════════════════════════════════════════════════════════════
// 模式 1: JD 详情页提取 (v0.1 现有逻辑)
// ═══════════════════════════════════════════════════════════════════

function extractBossJD() {
  /* 在 BOSS JD 详情页 context 里跑. 自包含, 不能引用 popup 作用域. */
  const url = window.location.href;

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
  const descSelectors = [
    ".job-detail .job-sec-text",
    ".job-sec-text",
    "[class*='job-detail-section']",
    ".text.fold-text",
    ".text",
  ];
  for (const sel of descSelectors) {
    const el = document.querySelector(sel);
    if (el && el.innerText.trim().length > 50) {
      description = el.innerText.trim();
      break;
    }
  }
  if (!description) {
    const main = document.querySelector("main") ||
                 document.querySelector(".job-detail") ||
                 document.body;
    description = main.innerText.substring(0, 5000).trim();
  }

  const tags = Array.from(
    document.querySelectorAll(".job-tags .tag-item, .tag-list .tag, [class*='job-tag']")
  ).map(el => el.textContent.trim()).filter(Boolean);

  return { url, title, company, salary, location, description, tags };
}

// ═══════════════════════════════════════════════════════════════════
// 模式 2: 推荐列表提取 (W15.18 新增)
// ═══════════════════════════════════════════════════════════════════

function extractBossList() {
  /* 在 BOSS 推荐 / 搜索 / 列表页 context 里跑.
     提取**当前页可见**的所有岗位卡片 → 数组. 不分页跟踪 (用户自己滚就提取一次).
     Selectors 是基于 BOSS 2026 DOM, 用 robust fallback chain. */

  function trySelectorList(selectors, root) {
    for (const sel of selectors) {
      const els = (root || document).querySelectorAll(sel);
      if (els.length > 0) return Array.from(els);
    }
    return [];
  }

  function text(el, sel) {
    if (!el) return "";
    const c = el.querySelector(sel);
    return c ? c.textContent.trim() : "";
  }

  // 找到所有岗位卡片. BOSS 推荐页常用 .job-card-wrapper / .job-list-box li / [data-jid]
  const cards = trySelectorList([
    ".job-list-box li.job-card-box",  // 推荐页
    ".job-list-container .job-card-wrapper",
    "li.job-card-wrapper",
    "[data-jid]",
    ".job-card",
    ".job-list .job-primary",
  ]);

  const items = [];
  for (const card of cards) {
    const title = text(card, "[class*='job-name']") ||
                  text(card, ".job-title") ||
                  text(card, ".name");

    // 公司名 — BOSS 上常嵌套
    const company = text(card, "[class*='company-name']") ||
                    text(card, ".company-text .name") ||
                    text(card, ".company-name");

    const salary = text(card, "[class*='salary']") ||
                   text(card, ".job-salary") ||
                   text(card, ".salary");

    // 地点 — 通常一坨 location/exp/edu, 取第一个
    const locText = text(card, "[class*='job-area']") ||
                    text(card, ".location") ||
                    text(card, ".job-pub-time");
    const location = locText.split(/[\n\s]+/)[0] || locText;

    // JD link — 用于后续 fetch 详情
    let jdUrl = "";
    const linkEl = card.querySelector("a[href*='/job_detail/']") ||
                   card.querySelector("a[ka^='job-name']") ||
                   card.querySelector("a");
    if (linkEl && linkEl.href) {
      jdUrl = linkEl.href;
    }

    // 标签 (通常 skill / 福利)
    const tags = Array.from(
      card.querySelectorAll("[class*='job-tag'], .tag-list li, .info-desc")
    ).map(el => el.textContent.trim()).filter(t => t && t.length < 30).slice(0, 8);

    if (title && company) {  // 必填: title + company
      items.push({
        url: jdUrl,
        title, company, salary, location,
        tags,
        // 列表模式 description 是空 — 后续可由后端 fetch 补
        description: "",
      });
    }
  }

  return {
    page_url: window.location.href,
    items,
    captured_at: new Date().toISOString(),
  };
}

// ═══════════════════════════════════════════════════════════════════
// 检测当前页 mode + 执行提取
// ═══════════════════════════════════════════════════════════════════

async function doExtract() {
  setStatus("info", "检测中…");
  fieldsDetail.classList.add("hidden");
  fieldsList.classList.add("hidden");
  btnSend.disabled = true;
  extractedData = null;
  currentMode = null;
  modePill.classList.add("hidden");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url || "";

    // ── mode 1: JD 详情页
    if (URL_PATTERNS.jd_detail.test(url)) {
      setMode("jd_detail", "JD 详情");
      setStatus("info", "提取 JD 详情中…");

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractBossJD,
      });
      if (!results || !results[0] || !results[0].result) {
        setStatus("err", "提取失败 — 页面可能没加载完, 重新提取试试");
        return;
      }
      extractedData = results[0].result;
      fTitle.textContent = extractedData.title || "(未检测到)";
      fCompany.textContent = extractedData.company || "(未检测到)";
      fSalary.textContent = extractedData.salary || "(未检测到)";
      fLocation.textContent = extractedData.location || "(未检测到)";
      fDesc.textContent = (extractedData.description || "").substring(0, 300) + "…";
      fTags.textContent = extractedData.tags.join(", ") || "(无)";
      btnSend.disabled = false;
      setStatus("ok", "已提取 — 确认后发送");
      return;
    }

    // ── mode 2: 推荐列表 (W15.18 新增)
    if (URL_PATTERNS.recommend_list.test(url)) {
      setMode("recommend_list", "推荐列表");
      setStatus("info", "提取列表中所有岗位…");

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractBossList,
      });
      if (!results || !results[0] || !results[0].result) {
        setStatus("err", "提取失败 — DOM 结构可能变了, 重新提取试试");
        return;
      }
      extractedData = results[0].result;
      const items = extractedData.items || [];

      if (items.length === 0) {
        setStatus("warn", "没找到岗位卡片 — 滚动让岗位加载后重新提取");
        return;
      }

      // 渲染列表
      listSummary.textContent = `找到 ${items.length} 个岗位 — 一键全部 sync 到 OfferGuide`;
      listItems.innerHTML = "";
      for (const item of items.slice(0, 30)) {  // 最多展示 30 个
        const div = document.createElement("div");
        div.className = "list-item";
        const safe = (s) => (s || "").replace(/[<>]/g, "");
        div.innerHTML = `
          <div class="title">${safe(item.title)}</div>
          <div class="meta">${safe(item.company)} · ${safe(item.salary || "?")} · ${safe(item.location || "?")}</div>
        `;
        listItems.appendChild(div);
      }
      if (items.length > 30) {
        const more = document.createElement("div");
        more.className = "list-item meta";
        more.textContent = `…还有 ${items.length - 30} 个 (全部会发送)`;
        listItems.appendChild(more);
      }
      btnSend.disabled = false;
      setStatus("ok", `已提取 ${items.length} 个岗位 — 确认后批量发送`);
      return;
    }

    // ── 不在支持的页面
    setStatus("warn",
      "当前页面不是 BOSS JD 详情 (zhipin.com/job_detail/…) 也不是推荐列表 (zhipin.com/web/geek/recommend 等). " +
      "请打开支持的 BOSS 页面后重新提取."
    );
  } catch (err) {
    setStatus("err", "提取出错: " + err.message);
  }
}

// ═══════════════════════════════════════════════════════════════════
// 发送到 OfferGuide 后端 — 按 mode 走不同 endpoint
// ═══════════════════════════════════════════════════════════════════

async function doSend() {
  if (!extractedData || !currentMode) return;
  btnSend.disabled = true;
  setStatus("info", "正在发送…");

  let endpoint = "";
  let body = null;

  if (currentMode === "jd_detail") {
    endpoint = "/api/extension/ingest";
    body = extractedData;
  } else if (currentMode === "recommend_list") {
    endpoint = "/api/extension/bulk_ingest";
    body = extractedData;
  } else {
    setStatus("err", "未知 mode, 重新提取");
    return;
  }

  try {
    const resp = await fetch(API_BASE + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      setStatus("err", `发送失败 (${resp.status}): ${txt.substring(0, 120)}`);
      btnSend.disabled = false;
      return;
    }
    const data = await resp.json();

    if (currentMode === "jd_detail") {
      if (data.is_new) {
        setStatus("ok", `已入库 (job #${data.job_id}) — 在 OfferGuide 看分析`);
      } else {
        setStatus("info", `该岗位已存在 (job #${data.job_id}), 跳过`);
      }
    } else {
      // bulk_ingest 返回: { inserted: N, duplicate: M, total: T, job_ids: [...] }
      const ins = data.inserted || 0;
      const dup = data.duplicate || 0;
      setStatus("ok",
        `发送成功 — 新入库 ${ins} 个 / 跳过重复 ${dup} 个. ` +
        `打开 http://localhost:8000/jobs 看跟踪列表.`
      );
    }
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      setStatus("err",
        "连接失败 — 请确认 OfferGuide 后端已启动 " +
        "(python -m offerguide.ui.web 或 ./install.sh 后跑起来)"
      );
    } else {
      setStatus("err", "发送出错: " + err.message);
    }
    btnSend.disabled = false;
  }
}

// ── event wiring ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", doExtract);
btnRefresh.addEventListener("click", doExtract);
btnSend.addEventListener("click", doSend);
