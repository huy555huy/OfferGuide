"""W21 Phase 4 — Evaluation tools + Evaluation sub-agent tests.

Wrap existing SKILLs as tools. Tests use stub SkillRuntime so no real LLM.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import offerguide
from offerguide.agents.base import register_universal_tools
from offerguide.agents.evaluation import EvaluationSubAgent
from offerguide.harness import _schema as harness_schema
from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.tools.registry import ToolRegistry


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "eval.db")
    s.init_schema()
    harness_schema.init_harness_schema(s)
    return s


@pytest.fixture
def reg_with_evaluation():
    """Fresh registry with evaluation tools copied from global."""
    r = ToolRegistry()
    register_universal_tools(r)
    import offerguide.tools.evaluation  # noqa: F401  # registers to global
    from offerguide.tools.registry import registry as global_reg
    for group in ("evaluation", "shared"):
        for name in global_reg.names_in_group(group):
            entry = global_reg.get(name)
            if entry is not None and r.get(name) is None:
                r.register(
                    name=entry.name, group=entry.group,
                    schema=entry.schema, handler=entry.handler,
                    description=entry.description,
                    deprecated_aliases=entry.deprecated_aliases,
                )
    return r


@pytest.fixture
def jobs_with_score(store):
    """Insert 2 jobs in DB."""
    ids = []
    with store.connect() as conn:
        for i in range(2):
            cur = conn.execute(
                "INSERT INTO jobs (source, title, company, location, url, "
                "raw_text, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("nowcoder", f"job{i}", f"co{i}", "上海",
                 f"https://example.com/{i}", "x" * 300, f"h{i}"),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    return ids


def _stub_skill(name, parsed_output):
    """Return a SkillSpec mock + runtime mock that returns parsed_output."""
    spec = MagicMock()
    spec.name = name
    runtime = MagicMock()

    result = MagicMock()
    result.parsed = parsed_output
    result.raw_text = json.dumps(parsed_output, ensure_ascii=False)
    result.skill_run_id = 42
    result.cost_usd = 0.0003
    result.skill_version = "0.2.0"

    runtime.invoke = MagicMock(return_value=result)
    return spec, runtime


# ── Tool registration sanity ──────────────────────────────────────


class TestEvalToolRegistration:
    def test_6_eval_tools_plus_read_job_registered(self, reg_with_evaluation):
        eval_names = reg_with_evaluation.names_in_group("evaluation")
        expected = {
            "score_job", "generate_apply_pack", "generate_interview_prep",
            "tailor_resume", "find_resume_gaps", "compare_jobs",
        }
        assert set(eval_names) == expected
        # read_job is shared
        assert reg_with_evaluation.get("read_job") is not None
        assert reg_with_evaluation.get("read_job").group == "shared"

    def test_score_job_has_alias_score_match(self, reg_with_evaluation):
        """Backwards-compat: old prompts can still call by alias."""
        entry = reg_with_evaluation.get("score_match")
        assert entry is not None
        assert entry.name == "score_job"  # resolved to canonical


# ── score_job ─────────────────────────────────────────────────────


class TestScoreJob:
    def test_score_job_returns_parsed_skill_output(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        spec, runtime = _stub_skill("score_match", {
            "probability": 0.75,
            "reasoning": "good fit",
            "dimensions": {"tech": 0.9, "exp": 0.7, "company_tier": 0.6},
            "deal_breakers": [],
        })

        out = reg_with_evaluation.dispatch(
            "score_job", {"job_id": jobs_with_score[0]},
            store=store, runtime=runtime, skills=[spec],
            user_profile_text="resume text " * 50,
        )
        d = json.loads(out)
        assert d["probability"] == 0.75
        assert d["dimensions"]["tech"] == 0.9
        assert d["_meta"]["skill_run_id"] == 42
        assert d["_meta"]["cost_usd"] == 0.0003

    def test_score_job_writes_scored_harness_event(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        """score_job tool MUST write harness_events.kind='scored' so
        /recommended view ranks it."""
        spec, runtime = _stub_skill("score_match", {
            "probability": 0.62, "reasoning": "ok", "dimensions": {},
            "deal_breakers": ["要 985"],
        })

        reg_with_evaluation.dispatch(
            "score_job", {"job_id": jobs_with_score[0]},
            store=store, runtime=runtime, skills=[spec],
            user_profile_text="resume",
        )
        with store.connect() as conn:
            row = conn.execute(
                "SELECT job_id, json_extract(note, '$.probability') "
                "FROM harness_events WHERE kind='scored'"
            ).fetchone()
        assert row is not None
        assert row[0] == jobs_with_score[0]
        assert float(row[1]) == 0.62

    def test_score_job_missing_user_profile_returns_error(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        out = reg_with_evaluation.dispatch(
            "score_job", {"job_id": jobs_with_score[0]},
            store=store, runtime=MagicMock(), skills=[MagicMock(name="score_match")],
            user_profile_text="",
        )
        d = json.loads(out)
        assert "error" in d
        assert "user_profile_text" in d["error"]

    def test_score_job_nonexistent_job_returns_error(
        self, store, reg_with_evaluation,
    ):
        out = reg_with_evaluation.dispatch(
            "score_job", {"job_id": 999},
            store=store, runtime=MagicMock(), skills=[],
            user_profile_text="x",
        )
        d = json.loads(out)
        assert "error" in d
        assert "not found" in d["error"]

    def test_score_job_runtime_missing_returns_error(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        out = reg_with_evaluation.dispatch(
            "score_job", {"job_id": jobs_with_score[0]},
            store=store, runtime=None, skills=[],
            user_profile_text="resume",
        )
        d = json.loads(out)
        assert "error" in d


# ── generate_apply_pack ───────────────────────────────────────────


class TestGenerateApplyPack:
    def test_apply_pack_links_skill_to_job_via_harness_event(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        spec, runtime = _stub_skill("apply_assistant", {
            "self_intro_snippet": {"text": "..."},
            "qa_templates": [],
            "submission_strategy": {},
        })
        reg_with_evaluation.dispatch(
            "generate_apply_pack", {"job_id": jobs_with_score[0]},
            store=store, runtime=runtime, skills=[spec],
            user_profile_text="resume",
        )
        # W20.5 multi-SKILL attribution: harness_event 'apply_assistant_generated'
        # written for this job
        with store.connect() as conn:
            row = conn.execute(
                "SELECT job_id, json_extract(note, '$.skill_name') "
                "FROM harness_events WHERE kind='apply_assistant_generated'"
            ).fetchone()
        assert row is not None
        assert row[0] == jobs_with_score[0]


# ── compare_jobs ─────────────────────────────────────────────────


class TestCompareJobs:
    def test_compare_jobs_needs_at_least_2(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        out = reg_with_evaluation.dispatch(
            "compare_jobs", {"job_ids": [jobs_with_score[0]]},
            store=store, runtime=MagicMock(), skills=[],
            user_profile_text="r",
        )
        d = json.loads(out)
        assert "error" in d

    def test_compare_jobs_invokes_skill_with_concat_jds(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        spec, runtime = _stub_skill("compare_jobs", {
            "winner": "job0", "rationale": "...",
        })
        out = reg_with_evaluation.dispatch(
            "compare_jobs", {"job_ids": jobs_with_score},
            store=store, runtime=runtime, skills=[spec],
            user_profile_text="r",
        )
        d = json.loads(out)
        assert d["winner"] == "job0"
        # Verify runtime.invoke was called with concatenated jobs
        call_args = runtime.invoke.call_args
        inputs = call_args[0][1]  # second positional arg = inputs dict
        assert "jobs_text" in inputs
        assert "job#" in inputs["jobs_text"]


# ── read_job (shared) ────────────────────────────────────────────


class TestReadJob:
    def test_read_job_returns_flat_dict(
        self, store, reg_with_evaluation, jobs_with_score,
    ):
        out = reg_with_evaluation.dispatch(
            "read_job", {"job_id": jobs_with_score[0]}, store=store,
        )
        d = json.loads(out)
        assert d["id"] == jobs_with_score[0]
        assert d["company"] == "co0"

    def test_read_job_not_found(self, store, reg_with_evaluation):
        out = reg_with_evaluation.dispatch(
            "read_job", {"job_id": 9999}, store=store,
        )
        d = json.loads(out)
        assert "error" in d


# ── EvaluationSubAgent end-to-end ────────────────────────────────


class TestEvalSubAgentEndToEnd:
    def test_sub_agent_calls_score_then_done(
        self, store, reg_with_evaluation, jobs_with_score, monkeypatch,
    ):
        spec, runtime = _stub_skill("score_match", {
            "probability": 0.8, "reasoning": "ok",
            "dimensions": {}, "deal_breakers": [],
        })

        responses = [
            LLMResponse(
                content="先 score 一下", model="stub", cost_usd=0.001,
                tool_calls=[ToolCall(
                    id="t1", name="score_job",
                    arguments={"job_id": jobs_with_score[0]},
                    arguments_raw=json.dumps({"job_id": jobs_with_score[0]}),
                )],
            ),
            LLMResponse(
                content="done", model="stub", cost_usd=0.0005,
                tool_calls=[ToolCall(
                    id="t2", name="done",
                    arguments={"summary": "score=0.8"},
                    arguments_raw='{"summary":"score=0.8"}',
                )],
            ),
        ]
        llm = MagicMock()
        llm.chat_with_tools = MagicMock(side_effect=responses)

        sub = EvaluationSubAgent(
            llm=llm, registry=reg_with_evaluation, store=store,
            runtime=runtime, skills=[spec],
            user_profile_text="resume " * 50,
            max_iter=4,
        )
        result = sub.run(goal=f"对 job#{jobs_with_score[0]} 打分")
        assert result.error is None
        assert "0.8" in result.final_answer

    def test_eval_sub_agent_sees_6_skills_plus_shared(
        self, store, reg_with_evaluation,
    ):
        captured = {}
        def _spy(messages, tools, temperature):
            captured["tools"] = tools
            return LLMResponse(content="ok", model="stub", tool_calls=[])
        llm = MagicMock()
        llm.chat_with_tools = _spy

        sub = EvaluationSubAgent(
            llm=llm, registry=reg_with_evaluation, store=store,
        )
        sub.run(goal="x")

        names = {t["function"]["name"] for t in captured["tools"]}
        # 6 evaluation tools + done (shared) + read_job (shared)
        assert "score_job" in names
        assert "generate_apply_pack" in names
        assert "generate_interview_prep" in names
        assert "tailor_resume" in names
        assert "find_resume_gaps" in names
        assert "compare_jobs" in names
        assert "done" in names
        assert "read_job" in names
        # Should NOT see discovery tools
        assert "fetch_nowcoder" not in names
