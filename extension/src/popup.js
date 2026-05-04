// Health check + counter display

async function checkHealth() {
  const el = document.getElementById("health-status");
  try {
    const r = await fetch("http://localhost:8000/api/extension/ping", {
      method: "GET",
    });
    if (r.ok) {
      el.textContent = "✓ ok";
      el.style.color = "#6b8e6b";
    } else {
      el.textContent = `HTTP ${r.status}`;
      el.style.color = "#b85f44";
    }
  } catch (e) {
    el.textContent = "未运行";
    el.style.color = "#b85f44";
  }
}

function refreshCount() {
  chrome.runtime.sendMessage({kind: "get_recent_count"}, (r) => {
    if (r) {
      document.getElementById("copy-count").textContent =
        `${r.lastMinute} (1m) / ${r.lastHour} (1h)`;
    }
  });
}

checkHealth();
refreshCount();
