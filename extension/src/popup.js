// Health check for the local OfferGuide service and wake the MV3 bridge worker.

async function checkHealth() {
  const el = document.getElementById("health-status");
  try {
    const r = await fetch("http://localhost:8000/api/extension/ping", {
      method: "GET",
    });
    if (r.ok) {
      el.textContent = "✓ ok";
      el.style.color = "#6b8e6b";
      chrome.runtime.sendMessage({kind: "wake_browser_bridge"}, () => {});
    } else {
      el.textContent = `HTTP ${r.status}`;
      el.style.color = "#b85f44";
    }
  } catch (e) {
    el.textContent = "未运行";
    el.style.color = "#b85f44";
  }
}

checkHealth();
