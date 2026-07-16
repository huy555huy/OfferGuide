from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from offerguide import Store
from offerguide.platforms.official_jobs import (
    BYTEDANCE_SEARCH_URL,
    TENCENT_CAMPUS_DETAIL_URL,
    TENCENT_CAMPUS_SEARCH_URL,
    TENCENT_SOCIAL_DETAIL_URL,
    TENCENT_SOCIAL_SEARCH_URL,
)
from offerguide.platforms.shixiseng import DETAIL_URL_TEMPLATE, LIST_URL
from offerguide.research_agents.job_discovery.models import JobPostingEvidence
from offerguide.research_agents.job_discovery.platform_tools import (
    build_platform_job_tools,
)
from offerguide.research_agents.job_discovery.source_tools import (
    SOURCE_STORE_DEPENDENCY,
    StoredJobSourceEvidenceVerifier,
)
from offerguide.research_agents.runner import (
    AgentExecutionContext,
    AgentSubjectContext,
    AgentToolResult,
)
from offerguide.research_agents.sources import SourceEvidenceStore


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Mapping[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = dict(payload or {})
        self.text = text

    def json(self) -> dict[str, Any]:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _ClientBase:
    calls: list[tuple[str, str, dict[str, Any]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _OfficialClient(_ClientBase):
    instances: ClassVar[list[_OfficialClient]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instances.append(self)

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> _Response:
        self.calls.append(("POST", url, dict(json)))
        assert url == TENCENT_CAMPUS_SEARCH_URL
        return _Response(
            payload={
                "status": 0,
                "data": {
                    "count": 1,
                    "positionList": [
                        {
                            "postId": "campus-1",
                            "positionTitle": "大模型产品实习生",
                            "projectName": "实习项目",
                            "bgs": "平台与内容事业群",
                        }
                    ],
                },
            }
        )

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> _Response:
        values = dict(params or {})
        self.calls.append(("GET", url, values))
        if url == TENCENT_CAMPUS_DETAIL_URL:
            assert values == {"postId": "campus-1"}
            return _Response(
                payload={
                    "status": 0,
                    "data": {
                        "postId": "campus-1",
                        "title": "大模型产品实习生",
                        "workCityList": ["深圳"],
                        "desc": "负责从用户问题发现到产品上线的完整过程。",
                        "request": "能够清楚分析需求并与研发协作。",
                    },
                }
            )
        if url == TENCENT_SOCIAL_SEARCH_URL:
            assert values["keyword"] == "Agent 产品"
            assert values["pageSize"] == 2
            return _Response(
                payload={
                    "Code": 200,
                    "Data": {"Count": 1, "Posts": [{"PostId": "social-1"}]},
                }
            )
        if url == TENCENT_SOCIAL_DETAIL_URL:
            assert values["postId"] == "social-1"
            return _Response(
                payload={
                    "Code": 200,
                    "Data": {
                        "PostId": "social-1",
                        "RecruitPostName": "Agent 产品经理",
                        "LocationName": "上海",
                        "PostURL": ("https://careers.tencent.com/jobdesc.html?postId=social-1"),
                        "Responsibility": "负责 Agent 产品规划和交付。",
                        "Requirement": "具备产品判断和跨团队协作能力。",
                    },
                }
            )
        raise AssertionError(f"unexpected official request: {url}")


@pytest.fixture
def evidence_store(tmp_path) -> SourceEvidenceStore:
    store = Store(tmp_path / "platform-tools.db")
    store.init_schema()
    evidence = SourceEvidenceStore(store)
    evidence.init_schema()
    return evidence


@pytest.fixture
def execution(evidence_store: SourceEvidenceStore) -> AgentExecutionContext:
    return AgentExecutionContext(
        run_id="platform-run",
        agent_name="JobDiscoveryAgent",
        subject=AgentSubjectContext(
            subject_kind="job_search",
            subject_id="current",
            subject_revision=3,
            result_revision=1,
            payload={},
        ),
        dependencies={SOURCE_STORE_DEPENDENCY: evidence_store},
        started_at=1.0,
        iteration=1,
        tool_call_id="tool-call",
    )


def _tool(name: str):
    return next(tool for tool in build_platform_job_tools() if tool.name == name)


def _content(result) -> dict[str, Any]:
    assert isinstance(result.content, dict)
    return result.content


def _invoke(
    name: str,
    args: Mapping[str, Any],
    execution: AgentExecutionContext,
) -> AgentToolResult:
    result = _tool(name).handler(args, execution)
    assert isinstance(result, AgentToolResult)
    return result


def _read_all(evidence_store: SourceEvidenceStore, evidence_id: int, page_size: int) -> str:
    chunks: list[str] = []
    offset = 0
    while True:
        page = evidence_store.read_page(
            evidence_id,
            offset=offset,
            max_chars=page_size,
        )
        chunks.append(page.content)
        if page.complete:
            assert page.next_offset is None
            break
        assert page.next_offset is not None
        offset = page.next_offset
    return "".join(chunks)


def test_official_tool_executes_real_adapter_requests_and_saves_complete_details(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _OfficialClient.instances.clear()
    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _OfficialClient,
    )

    result = _invoke(
        "search_verified_official_jobs",
        {"keyword": "Agent 产品", "company": "腾讯", "limit_per_source": 2},
        execution,
    )

    assert result.is_error is False
    content = _content(result)
    verified = content["verified_jobs"]
    assert [item["source_name"] for item in verified] == [
        "tencent_campus",
        "tencent_social",
    ]
    assert all(item["evidence_kind"] == "platform_adapter" for item in verified)
    assert all(item["platform_job_ref"].startswith("platform-job:") for item in verified)
    assert "完整过程" in verified[0]["jd_text"]
    assert "跨团队协作" in verified[1]["jd_text"]
    calls = _OfficialClient.instances[0].calls
    assert calls[0] == (
        "POST",
        TENCENT_CAMPUS_SEARCH_URL,
        {"keyword": "Agent 产品", "pageIndex": 1, "pageSize": 2},
    )
    assert {call[1] for call in calls} == {
        TENCENT_CAMPUS_SEARCH_URL,
        TENCENT_CAMPUS_DETAIL_URL,
        TENCENT_SOCIAL_SEARCH_URL,
        TENCENT_SOCIAL_DETAIL_URL,
    }

    history = evidence_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert [(item.backend, item.status) for item in history] == [
        ("tencent_campus", "succeeded"),
        ("tencent_social", "succeeded"),
    ]
    # Search rows are traceable clues, never evidence identities.
    assert all("evidence_id" not in clue for item in history for clue in item.results)
    attached = evidence_store.attached_evidence(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert len(attached) == 2
    assert {item.id for item in attached} == {item["source_evidence_id"] for item in verified}
    assert all(item.provenance == "web" for item in attached)
    assert all(item.text_content == item.raw_content.decode("utf-8") for item in attached)
    assert verified[0]["jd_text"] not in attached[0].text_content


def test_shared_platform_response_is_saved_once_and_reusable_for_two_jobs(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [
        {
            "id": "byte-1",
            "title": "Agent 产品经理",
            "description": "负责从真实用户问题发现到 Agent 产品方案上线的完整规划。",
            "requirement": "能够分析复杂需求并推动设计、工程和业务团队共同交付。",
            "city_info": {"name": "上海"},
            "recruit_type": {"name": "正式"},
        },
        {
            "id": "byte-2",
            "title": "大模型应用产品经理",
            "description": "负责从场景调研到大模型应用上线和效果复盘的完整交付。",
            "requirement": "能够拆解用户问题并与多个团队协作持续验证和改进方案。",
            "city_info": {"name": "北京"},
            "recruit_type": {"name": "正式"},
        },
    ]

    class _SharedResponseClient(_ClientBase):
        def post(
            self,
            url: str,
            *,
            json: Mapping[str, Any],
            headers: Mapping[str, str] | None = None,
        ) -> _Response:
            assert url == BYTEDANCE_SEARCH_URL
            return _Response(
                payload={
                    "message": "ok",
                    "data": {"count": 2, "job_post_list": jobs},
                }
            )

    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _SharedResponseClient,
    )
    result = _invoke(
        "search_verified_official_jobs",
        {"keyword": "Agent", "company": "字节", "limit_per_source": 2},
        execution,
    )

    assert result.is_error is False
    verified = _content(result)["verified_jobs"]
    assert len(verified) == 2
    evidence_ids = {item["source_evidence_id"] for item in verified}
    assert len(evidence_ids) == 1
    evidence_id = evidence_ids.pop()
    stored = evidence_store.get(evidence_id)
    assert stored.text_content == stored.raw_content.decode("utf-8")
    assert all(item["jd_text"] not in stored.text_content for item in verified)
    assert "byte-1" in stored.text_content
    assert "byte-2" in stored.text_content

    verifier = StoredJobSourceEvidenceVerifier(evidence_store)
    for item in verified:
        verifier.verify_platform_job_page(
            JobPostingEvidence.model_validate(
                {
                    key: value
                    for key, value in item.items()
                    if key in JobPostingEvidence.model_fields
                }
            ),
            subject_kind="job_search",
            subject_id="current",
            subject_revision=3,
        )


def test_official_adapter_metadata_without_role_body_is_not_verified_evidence(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ThinOfficialClient(_ClientBase):
        def post(
            self,
            url: str,
            *,
            json: Mapping[str, Any],
            headers: Mapping[str, str] | None = None,
        ) -> _Response:
            assert url == BYTEDANCE_SEARCH_URL
            return _Response(
                payload={
                    "message": "ok",
                    "data": {
                        "count": 1,
                        "job_post_list": [
                            {
                                "id": "thin-1",
                                "title": "Only a result card",
                                "description": "",
                                "requirement": "",
                                "city_info": {"name": "Shanghai"},
                            }
                        ],
                    },
                }
            )

    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _ThinOfficialClient,
    )
    result = _invoke(
        "search_verified_official_jobs",
        {"keyword": "result card", "company": "字节", "limit_per_source": 2},
        execution,
    )

    assert result.is_error is True
    assert _content(result)["verified_jobs"] == []
    assert _content(result)["source_attempts"][0]["status"] == "failed"
    assert evidence_store.attached_evidence(
        subject_kind="job_search", subject_id="current", subject_revision=3
    ) == ()


def test_shixiseng_tool_requests_list_and_detail_then_supports_complete_paged_read(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "岗位职责：" + "参与真实产品交付；" * 80 + "任职要求：能够分析用户问题。"
    token = "inn_abcdefgh"

    class _ShixisengClient(_ClientBase):
        instance: _ShixisengClient | None = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).instance = self

        def get(self, url: str, params: Mapping[str, Any] | None = None) -> _Response:
            values = dict(params or {})
            self.calls.append(("GET", url, values))
            if url == LIST_URL:
                return _Response(text=f'<a href="/intern/{token}">岗位</a>')
            if url == DETAIL_URL_TEMPLATE.format(token=token):
                return _Response(
                    text=(
                        "<title>Agent产品实习招聘-测试公司实习生招聘-实习僧</title>"
                        '<span title="上海" class="job_position">上海</span>'
                        '<span class="cutom_font">2026-07-14 09:00:00</span>'
                        f'<div class="job_detail">{body}</div>'
                    )
                )
            raise AssertionError(f"unexpected shixiseng request: {url}")

    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _ShixisengClient,
    )
    result = _invoke(
        "search_shixiseng_jobs",
        {"keyword": "Agent 产品", "page": 3, "limit": 1},
        execution,
    )

    assert result.is_error is False
    verified = _content(result)["verified_jobs"]
    assert len(verified) == 1
    assert verified[0]["canonical_url"] == DETAIL_URL_TEMPLATE.format(token=token)
    assert body in verified[0]["jd_text"]
    assert _ShixisengClient.instance is not None
    assert _ShixisengClient.instance.calls == [
        ("GET", LIST_URL, {"keyword": "Agent 产品", "page": 3}),
        ("GET", DETAIL_URL_TEMPLATE.format(token=token), {}),
    ]

    history = evidence_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert len(history) == 1
    assert history[0].status == "succeeded"
    assert history[0].results[0]["url"] == DETAIL_URL_TEMPLATE.format(token=token)
    assert "evidence_id" not in history[0].results[0]
    evidence_id = verified[0]["source_evidence_id"]
    stored = evidence_store.get(evidence_id)
    reconstructed = _read_all(evidence_store, evidence_id, page_size=37)
    assert reconstructed == stored.text_content
    assert body in reconstructed


def test_shixiseng_clue_does_not_become_evidence_when_detail_has_no_jd(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "inn_emptybody"

    class _MissingDetailClient(_ClientBase):
        def get(self, url: str, params: Mapping[str, Any] | None = None) -> _Response:
            if url == LIST_URL:
                return _Response(text=f'<a href="/intern/{token}">岗位 clue</a>')
            return _Response(text="<title>产品实习招聘-测试公司实习生招聘-实习僧</title>")

    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _MissingDetailClient,
    )
    result = _invoke(
        "search_shixiseng_jobs",
        {"keyword": "产品", "page": 1, "limit": 1},
        execution,
    )

    assert result.is_error is True
    assert _content(result)["verified_jobs"] == []
    history = evidence_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert history[0].status == "succeeded"
    assert len(history[0].results) == 1
    assert (
        evidence_store.attached_evidence(
            subject_kind="job_search", subject_id="current", subject_revision=3
        )
        == ()
    )
    failures = evidence_store.fetch_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert len(failures) == 1
    assert failures[0].status == "failed"
    assert failures[0].error_code == "empty_job_detail"


def test_restricted_official_source_is_failed_not_an_empty_success(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _ClientBase,
    )
    result = _invoke(
        "search_verified_official_jobs", {"keyword": "产品", "company": "美团"}, execution
    )

    assert result.is_error is True
    attempt = _content(result)["source_attempts"][0]
    assert attempt["adapter_status"] == "login_required"
    assert attempt["status"] == "failed"
    history = evidence_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert len(history) == 1
    assert history[0].status == "failed"
    assert "登录" in (history[0].error_text or "")


@pytest.mark.parametrize(
    ("tool_name", "args", "message"),
    [
        (
            "search_verified_official_jobs",
            {"keyword": "产品", "limit_per_source": 0},
            "limit_per_source must be between 1 and 20",
        ),
        (
            "search_shixiseng_jobs",
            {"keyword": "产品", "page": "not-an-integer"},
            "page must be an integer",
        ),
    ],
)
def test_platform_tools_reject_invalid_numeric_arguments_without_requesting(
    tool_name: str,
    args: dict[str, Any],
    message: str,
    execution: AgentExecutionContext,
) -> None:
    result = _invoke(tool_name, args, execution)
    assert result.is_error is True
    assert _content(result) == {"error": message}


def test_shixiseng_list_request_failure_is_persisted_and_returned_as_error(
    evidence_store: SourceEvidenceStore,
    execution: AgentExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingClient(_ClientBase):
        def get(self, url: str, params: Mapping[str, Any] | None = None) -> _Response:
            raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        "offerguide.research_agents.job_discovery.platform_tools.httpx.Client",
        _FailingClient,
    )
    result = _invoke(
        "search_shixiseng_jobs",
        {"keyword": "产品"},
        execution,
    )

    assert result.is_error is True
    assert "network unavailable" in _content(result)["error"]
    history = evidence_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=3
    )
    assert len(history) == 1
    assert history[0].status == "failed"
    assert "network unavailable" in (history[0].error_text or "")
