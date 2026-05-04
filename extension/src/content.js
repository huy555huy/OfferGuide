/* OfferGuide Helper — content script (W13.8)
 *
 * Injected into Boss直聘 / 牛客 pages. Inserts a small floating panel that:
 *   1. Detects which company the page is about (from page title / URL)
 *   2. Asks the OfferGuide backend (via background.js) for an apply package
 *   3. Renders ONE button per package piece (self-intro, each Q/A) — each
 *      button only writes to clipboard, NEVER touches platform DOM
 *
 * Anti-ban design:
 *   - We NEVER call any platform DOM API beyond reading visible text
 *     (page title for company sniffing). No mutation observers, no event
 *     dispatch, no programmatic clicks.
 *   - Every "copy" is user-triggered (real mousedown on our button)
 *   - Background rate-limits 5 copies/min / 30/hour
 *   - Floating panel is visually obvious (orange 🛠 icon) so users can
 *     dismiss / hide it. Not stealthy.
 */

(function() {
  "use strict";

  if (window.__offerguide_helper_loaded) return;
  window.__offerguide_helper_loaded = true;

  // ────────── Detect company name from page title / URL ──────────
  function sniffCompany() {
    const title = document.title || "";
    // Boss直聘 patterns:
    //   "<招聘官 ID> | <职位> | <公司> | BOSS直聘"
    //   "<职位>招聘 - <公司> - BOSS直聘"
    const bossMatch = title.match(/[|｜\-]\s*([一-龥A-Za-z0-9.]{2,20}(?:科技|公司|集团|有限|股份|网络|跳动|腾讯|阿里|百度|美团|小红书)?)\s*[|｜\-]/);
    if (bossMatch) return bossMatch[1].trim();
    // Niuke (NowCoder) patterns:
    //   "<公司>招聘 - 牛客网"
    const nkMatch = title.match(/^(.+?)招聘.*牛客/);
    if (nkMatch) return nkMatch[1].trim();
    return null;
  }

  // ────────── Build floating panel ──────────
  function buildPanel() {
    const panel = document.createElement("div");
    panel.id = "offerguide-helper-panel";
    panel.innerHTML = `
      <div class="og-header">
        <span class="og-icon">🛠</span>
        <span class="og-title">OfferGuide</span>
        <button class="og-close" title="收起">×</button>
      </div>
      <div class="og-body">
        <div class="og-status">检测公司中...</div>
        <div class="og-content"></div>
      </div>
      <div class="og-footer">
        🔒 仅复制到剪贴板, 永不替你点提交 · <span class="og-rate"></span>
      </div>
    `;
    document.documentElement.appendChild(panel);

    panel.querySelector(".og-close").addEventListener("click", () => {
      panel.classList.toggle("og-collapsed");
    });
    return panel;
  }

  // ────────── Render a copy button ──────────
  function makeCopyBtn(label, text, hint) {
    const btn = document.createElement("button");
    btn.className = "og-copy-btn";
    btn.innerHTML = `<span class="og-copy-label">${label}</span><span class="og-copy-action">复制</span>`;
    if (hint) btn.title = hint;

    btn.addEventListener("click", async () => {
      // Check rate limit BEFORE writing
      const rate = await new Promise(r => chrome.runtime.sendMessage({kind: "check_rate"}, r));
      if (!rate.ok) {
        btn.querySelector(".og-copy-action").textContent = "⚠ 太快";
        btn.title = `节流: ${rate.reason}. 慢一点, 平台风控敏感。`;
        setTimeout(() => {
          btn.querySelector(".og-copy-action").textContent = "复制";
          btn.title = hint || "";
        }, 3000);
        return;
      }

      try {
        await navigator.clipboard.writeText(text);
        chrome.runtime.sendMessage({kind: "record_copy"});
        btn.classList.add("og-copied");
        btn.querySelector(".og-copy-action").textContent = "✓ 已复制";
        // Refresh rate counter shown in footer
        chrome.runtime.sendMessage({kind: "get_recent_count"}, (r) => {
          const rateEl = document.querySelector(".og-rate");
          if (rateEl && r) {
            rateEl.textContent = `本小时 ${r.lastHour} 次复制`;
          }
        });
        setTimeout(() => {
          btn.classList.remove("og-copied");
          btn.querySelector(".og-copy-action").textContent = "复制";
        }, 2000);
      } catch (err) {
        btn.querySelector(".og-copy-action").textContent = "✗ 失败";
        console.warn("OfferGuide clipboard write failed:", err);
      }
    });
    return btn;
  }

  // ────────── Render package into panel ──────────
  function renderPackage(panel, pkg, jobId) {
    const body = panel.querySelector(".og-content");
    body.innerHTML = "";

    if (pkg.skip_reasons && pkg.skip_reasons.length > 0) {
      const warn = document.createElement("div");
      warn.className = "og-skip-warn";
      warn.innerHTML = `⚠ 不建议投: ${pkg.skip_reasons.join(" · ")}`;
      body.appendChild(warn);
    }

    // Self intro
    if (pkg.self_intro_snippet?.text) {
      body.appendChild(makeSection("📨 第一句话",
        [makeCopyBtn("自我介绍", pkg.self_intro_snippet.text, pkg.self_intro_snippet.rationale)]
      ));
    }

    // Q/A
    if (pkg.qa_templates?.length) {
      const qaButtons = pkg.qa_templates.map((qa, i) =>
        makeCopyBtn(`Q${i + 1}: ${qa.question.slice(0, 30)}`, qa.answer,
                    `category=${qa.category} · 个性化 ${qa.personalization_score}`)
      );
      body.appendChild(makeSection("📝 表单问答", qaButtons));
    }

    // Strategy reminders (text only, no copy needed)
    if (pkg.submission_strategy) {
      const s = pkg.submission_strategy;
      const stratDiv = document.createElement("div");
      stratDiv.className = "og-strategy";
      stratDiv.innerHTML = `
        <div class="og-section-title">🎯 投递策略</div>
        <div class="og-strat-item">最佳时间: ${s.best_time_window || "—"}</div>
        <div class="og-strat-item">预期回复: ${s.expected_response_window_days || "?"} 天</div>
      `;
      body.appendChild(stratDiv);
    }

    // Job tracking link back to OfferGuide
    if (jobId) {
      const link = document.createElement("a");
      link.className = "og-track-link";
      link.href = `http://localhost:8000/apply/${jobId}`;
      link.target = "_blank";
      link.textContent = `→ 在 OfferGuide 标记投递状态 (job#${jobId})`;
      body.appendChild(link);
    }
  }

  function makeSection(title, buttons) {
    const sec = document.createElement("div");
    sec.className = "og-section";
    sec.innerHTML = `<div class="og-section-title">${title}</div>`;
    const btnRow = document.createElement("div");
    btnRow.className = "og-btn-row";
    buttons.forEach(b => btnRow.appendChild(b));
    sec.appendChild(btnRow);
    return sec;
  }

  // ────────── Main ──────────
  function main() {
    const company = sniffCompany();
    const panel = buildPanel();
    const status = panel.querySelector(".og-status");

    if (!company) {
      status.textContent = "未识别到公司名 (在岗位详情页或聊天页才有)";
      return;
    }
    status.innerHTML = `检测到: <strong>${company}</strong> · 拉取投递包...`;

    chrome.runtime.sendMessage(
      {kind: "fetch_package_for_company", company},
      (resp) => {
        if (!resp || !resp.ok) {
          status.innerHTML = `${company}: 没找到投递包<br>
            <span class="og-hint">先去 <a href="http://localhost:8000/agent" target="_blank">OfferGuide /agent</a> 跑 apply_assistant 准备一份</span>`;
          return;
        }
        status.innerHTML = `<strong>${company}</strong> 投递包已就绪`;
        renderPackage(panel, resp.package, resp.job_id);
      },
    );
  }

  // Run after page settles
  if (document.readyState === "complete") {
    setTimeout(main, 500);
  } else {
    window.addEventListener("load", () => setTimeout(main, 500));
  }
})();
