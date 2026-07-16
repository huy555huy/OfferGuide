/* OfferGuide Helper — content script
 *
 * Injected into Boss直聘 / 牛客 pages. Inserts a small floating panel that:
 *   1. Detects which company the page is about (from page title / URL)
 *   2. Asks the OfferGuide backend (via background.js) for an apply package
 *   3. Renders one button for the message and each form answer; each
 *      button only writes to clipboard, NEVER touches platform DOM
 *
 * Anti-ban design:
 *   - We NEVER call any platform DOM API beyond reading visible text
 *     (page title for company sniffing). No mutation observers, no event
 *     dispatch, no programmatic clicks.
 *   - Every "copy" is user-triggered (real mousedown on our button)
 *   - The floating panel is visible and can be collapsed by the user.
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
        仅复制到剪贴板；发送或提交前由你检查并确认
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
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("og-copied");
        btn.querySelector(".og-copy-action").textContent = "✓ 已复制";
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

    if (pkg.message) {
      body.appendChild(makeSection("📨 第一句话",
        [makeCopyBtn("沟通话术", pkg.message, "当前岗位投递包中的沟通文案")]
      ));
    }

    if (pkg.form_answers?.length) {
      const qaButtons = pkg.form_answers.map((qa, i) =>
        makeCopyBtn(`Q${i + 1}: ${qa.question.slice(0, 30)}`, qa.answer, qa.question)
      );
      body.appendChild(makeSection("📝 表单问答", qaButtons));
    }

    if (pkg.pre_submit_checks?.length) {
      const checks = document.createElement("div");
      checks.className = "og-strategy";
      const title = document.createElement("div");
      title.className = "og-section-title";
      title.textContent = "提交前检查";
      checks.appendChild(title);
      pkg.pre_submit_checks.forEach((item) => {
        const row = document.createElement("div");
        row.className = "og-strat-item";
        row.textContent = item;
        checks.appendChild(row);
      });
      body.appendChild(checks);
    }

    // Job tracking link back to OfferGuide
    if (jobId) {
      const link = document.createElement("a");
      link.className = "og-track-link";
      link.href = `http://localhost:8000/jobs/${jobId}/apply-pack`;
      link.target = "_blank";
      link.textContent = `→ 查看 OfferGuide 投递包 (job#${jobId})`;
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

  function fetchPackage(company, jobId, callback) {
    chrome.runtime.sendMessage(
      {kind: "fetch_package_for_company", company, job_id: jobId || null},
      callback,
    );
  }

  function renderWorkspaceChoices(panel, company, matches) {
    const body = panel.querySelector(".og-content");
    body.innerHTML = "";
    const buttons = matches.map((match) => {
      const button = document.createElement("button");
      button.className = "og-copy-btn";
      const label = document.createElement("span");
      label.className = "og-copy-label";
      label.textContent = match.title;
      const action = document.createElement("span");
      action.className = "og-copy-action";
      action.textContent = "选择";
      button.append(label, action);
      button.addEventListener("click", () => {
        fetchPackage(company, match.job_id, (response) => {
          if (!response?.ok) {
            panel.querySelector(".og-status").textContent = response?.error || "投递包读取失败";
            return;
          }
          panel.querySelector(".og-status").textContent = `${company} · ${match.title}`;
          renderPackage(panel, response.package, response.job_id);
        });
      });
      return button;
    });
    body.appendChild(makeSection("选择当前岗位", buttons));
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

    fetchPackage(company, null, (resp) => {
        if (resp?.status === 409 && resp.matches?.length) {
          status.textContent = `${company}: 请选择当前岗位`;
          renderWorkspaceChoices(panel, company, resp.matches);
          return;
        }
        if (!resp || !resp.ok) {
          status.innerHTML = `${company}: 没找到当前投递包<br>
            <span class="og-hint">先去 <a href="http://localhost:8000/pipeline" target="_blank">OfferGuide Pipeline</a> 选择这个岗位并打开投递包</span>`;
          return;
        }
        status.innerHTML = `<strong>${company}</strong> 投递包已就绪`;
        renderPackage(panel, resp.package, resp.job_id);
      });
  }

  // Run after page settles
  if (document.readyState === "complete") {
    setTimeout(main, 500);
  } else {
    window.addEventListener("load", () => setTimeout(main, 500));
  }
})();
