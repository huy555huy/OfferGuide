"""飞书 (Feishu / Lark) custom-bot webhook notifier.

The webhook URL has shape `https://open.feishu.cn/open-apis/bot/v2/hook/<TOKEN>`.
Body: `{"msg_type":"text","content":{"text":"..."}}` for plain text;
`{"msg_type":"interactive","card":{...}}` for cards (W5+ when we want richer
formatting). For W4 we send `text` with title and body concatenated — works
with any feishu group bot regardless of token-level permissions.

We intentionally do NOT support feishu's signature verification in W4: it adds
HMAC plumbing for one feature (replay protection on the inbound webhook URL)
that's only meaningful for public webhook listeners, not for outbound posts.
"""

from __future__ import annotations

import httpx

from ._base import NotifyLevel, NotifyResult

_LEVEL_PREFIX = {"info": "📨", "warn": "⚠️", "high": "🔥"}


class FeishuNotifier:
    name = "feishu"

    def __init__(self, webhook_url: str, *, timeout_s: float = 10.0) -> None:
        if not webhook_url:
            raise ValueError("FeishuNotifier needs a webhook URL")
        self._url = webhook_url
        self._http = httpx.Client(timeout=timeout_s)

    def notify(
        self,
        *,
        title: str,
        body: str,
        level: NotifyLevel = "info",
    ) -> NotifyResult:
        prefix = _LEVEL_PREFIX.get(level, "📨")
        payload = {
            "msg_type": "text",
            "content": {"text": f"{prefix} {title}\n\n{body}"},
        }
        try:
            resp = self._http.post(self._url, json=payload)
        except httpx.HTTPError as e:
            return NotifyResult(ok=False, channel=self.name, error=str(e))
        if resp.status_code != 200:
            return NotifyResult(
                ok=False,
                channel=self.name,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        try:
            data = resp.json()
        except Exception as e:  # pragma: no cover — feishu always returns JSON
            return NotifyResult(ok=False, channel=self.name, error=f"non-JSON: {e}")
        # 飞书 returns {"code": 0, "msg": "ok"} on success; non-zero code is a logical error
        if data.get("code", 0) != 0:
            return NotifyResult(
                ok=False, channel=self.name, error=data.get("msg") or str(data), raw=data
            )
        return NotifyResult(ok=True, channel=self.name, raw=data)

    def close(self) -> None:
        self._http.close()
