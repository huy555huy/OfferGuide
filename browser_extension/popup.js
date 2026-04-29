/**
 * OfferGuide Boss 助手 — popup script.
 *
 * Flow: popup opens → check active tab URL → if Boss JD page, inject
 * extraction function via chrome.scripting.executeScript → show
 * extracted fields for review → user clicks "发送到 OfferGuide" →
 * POST to local FastAPI endpoint.
 *
 * Design choice: no content_script, no background service-worker.
 * Extraction only runs when the user clicks the extension icon —
 * "默认不自动发送" per the strategy memo.
 */

/* global chrome */

const API_BASE = "http://localhost:8000";
const BOSS_PATTERN = /^https?:\/\/(www\.)?zhipin\.com\/job_detail\//;

// ── DOM refs ───────────────────────────────────────────────────────

const statusEl   = document.getElementById("status");
const fieldsEl   = document.getElementById("fields");
const btnSend    = document.getElementById("btn-send");
const btnRefresh = document.getElementById("btn-refresh");
const fTitle     = document.getElementById("f-title");
const fCompany   = document.getElementById("f-company");
const fSalary    = document.getElementById("f-salary");
const fLocation  = document.getElementById("f-location");
const fDesc      = document.getElementById("f-desc");
const fTags      = document.getElementById("f-tags");

let extractedData = null;

// ── status helpers ─────────────────────────────────────────────────

function setStatus(cls, msg) {
  statusEl.className = "status " + cls;
  statusEl.textContent = msg;
}

// ── extraction function (runs IN the Boss page context) ────────────

function extractBossJD() {
  /* This function is serialized and injected into the active tab.
     It must be self-contained — no closures over popup scope. */
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

  // Job description — multiple fallback selectors
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

// ── extract from active tab ────────────────────────────────────────

async function doExtract() {
  setStatus("info", "检测中…");
  fieldsEl.classList.add("hidden");
  btnSend.disabled = true;
  extractedData = null;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !BOSS_PATTERN.test(tab.url || "")) {
      setStatus("warn", "当前页面不是 Boss直聘 JD 页面（zhipin.com/job_detail/…）");
      return;
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractBossJD,
    });

    if (!results || !results[0] || !results[0].result) {
      setStatus("err", "提取失败——页面可能尚未加载完毕，请稍后重试");
      return;
    }

    extractedData = results[0].result;
    fTitle.textContent    = extractedData.title || "(未检测到)";
    fCompany.textContent  = extractedData.company || "(未检测到)";
    fSalary.textContent   = extractedData.salary || "(未检测到)";
    fLocation.textContent = extractedData.location || "(未检测到)";
    fDesc.textContent     = (extractedData.description || "").substring(0, 300) + "…";
    fTags.textContent     = extractedData.tags.join(", ") || "(无)";

    fieldsEl.classList.remove("hidden");
    btnSend.disabled = false;
    setStatus("ok", "已提取——请确认后点击「发送到 OfferGuide」");
  } catch (err) {
    setStatus("err", "提取出错: " + err.message);
  }
}

// ── send to local OfferGuide backend ───────────────────────────────

async function doSend() {
  if (!extractedData) return;
  btnSend.disabled = true;
  setStatus("info", "正在发送…");

  try {
    const resp = await fetch(API_BASE + "/api/extension/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(extractedData),
    });

    if (!resp.ok) {
      const body = await resp.text();
      setStatus("err", `发送失败 (${resp.status}): ${body.substring(0, 120)}`);
      btnSend.disabled = false;
      return;
    }

    const data = await resp.json();
    if (data.is_new) {
      setStatus("ok", `已入库 (job #${data.job_id})——可在 OfferGuide 中分析`);
    } else {
      setStatus("info", `该岗位已存在 (job #${data.job_id})，无需重复导入`);
    }
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      setStatus("err", "连接失败——请确认 OfferGuide 后端已启动 (python -m offerguide.ui.web)");
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
