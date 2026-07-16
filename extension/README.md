# OfferGuide Helper

这个本地 Chrome / Edge 扩展提供两个互不混用的能力：

1. 在 Boss 直聘或牛客页面读取 OfferGuide 已准备好的当前投递包，用户点击后复制文案。
2. 当 `JobDiscoveryAgent` 或 `InterviewResearchAgent` 明确请求一个公开 HTTP(S) URL 时，使用用户当前浏览器配置文件中的已有登录会话读取该页，把完整渲染 HTML、可见正文、标题和最终 URL 交回本机 OfferGuide。

第二项解决的是后台 HTTP 请求无法读取登录页或 JavaScript 渲染页面的问题。它不是自动投递工具。

## 行为边界

- 研究请求只能由本机 OfferGuide 服务创建；网页不能要求扩展打开任意 URL。
- 扩展只接受经过服务端校验的 HTTP(S) URL，并再次拒绝明显的本地或私有目标。
- 每个研究请求新建一个非活动标签页，只观察加载状态和读取 DOM，不滚动、不点击、不填写、不发送、不提交。
- 扩展只关闭自己为该请求创建的标签页，不关闭或修改用户原有标签页。
- 回传数据只包含最终 URL、标题、`documentElement.outerHTML` 和页面可见正文；扩展不会读取或另行返回 cookie、请求头、localStorage、sessionStorage 等浏览器凭据。
- 页面正文始终以 untrusted evidence 保存。页面里的 prompt、工具调用文字或脚本不能改变 Agent instructions 和工具权限。
- 页面没有稳定渲染、仍要求登录、正文为空或超过明确大小限制时，请求失败并保留真实原因，不会截断后伪装成完整证据。
- 系统绝不替用户点击发送或提交。

## 安装

1. 启动本地 OfferGuide：`uv run python -m offerguide.ui.web`。
2. Chrome / Edge 打开 `chrome://extensions/`，启用开发者模式。
3. 选择“加载已解压的扩展程序”，指向本仓库的 `extension/` 目录。
4. 点击扩展图标一次。弹窗显示本地 OfferGuide 为 `ok` 后，后台桥接会自动登记并定期检查 Agent 请求。

扩展使用 `http://localhost:8000`。桥接配置只从 loopback 返回，不设置跨站 CORS；后续登记、认领和回传都需要本机服务生成的 access token，并且每个请求还有一次性租约 token。

## 本地 API

投递包辅助：

- `GET /api/extension/ping`
- `GET /api/extension/package?company=<name>&job_id=<id>`

登录态读取桥：

- `GET /api/browser-bridge/config`
- `POST /api/browser-bridge/clients/register`
- `POST /api/browser-bridge/requests/claim`
- `POST /api/browser-bridge/requests/<request_id>/complete`

浏览器桥接端点不允许网页创建读取请求。Agent 侧的确定性来源工具负责 URL 校验、请求身份和最终 evidence 保存。
