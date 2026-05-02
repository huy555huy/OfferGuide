"""Render and dispatch a SKILL — turns a `SkillSpec` + inputs into an LLM call.

Design choice: a SKILL's `body` IS the system prompt. Inputs are not interpolated
into the body — that would require a template language and tempt people to
hide logic in jinja conditions. Instead the runtime renders inputs as a
clearly labelled user message right after the system prompt:

    system: <SkillSpec.body>           # the instructions, shaped by the author / GEPA
    user:   "## 输入\n<labelled inputs>\n\n## 输出 (严格 JSON)\n"

Output: the LLM's raw text reply, parsed as JSON when `json_mode=True`. The
runtime records every call into `skill_runs` so the W6 GEPA evolver can build
its trainset directly from the user's real dogfood data.

Input handling (W5' fix): when a SKILL declares `inputs`, the runtime
canonicalizes the call to those keys exactly — extra keys raise (configurable),
missing keys raise. The same canonical dict drives **render**, **hash**, and
**persistence**, so an identical (spec, inputs) pair always produces the same
input_hash and the same skill_runs.input_json. Previously these three paths
disagreed when an extra key was passed: render skipped it, hash included it,
and persist serialized everything verbatim — meaning two functionally-equal
calls could be hashed differently and break GEPA's trainset deduplication.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient
from ..memory import Store
from ._spec import SkillSpec


@dataclass
class SkillResult:
    raw_text: str
    """The LLM's complete textual reply — always preserved, even when JSON parsing succeeded."""

    parsed: dict[str, Any] | None
    """JSON-decoded result when `json_mode=True` and parsing succeeded; else None."""

    skill_name: str
    skill_version: str
    skill_run_id: int
    """Row id in `skill_runs`. Lets feedback rows reference this specific call."""

    input_hash: str
    cost_usd: float
    latency_ms: int


class SkillRuntime:
    """Stateless dispatcher — `LLMClient` and `Store` are injected so tests can swap stubs."""

    def __init__(self, llm: LLMClient, store: Store) -> None:
        self._llm = llm
        self._store = store

    def invoke(
        self,
        spec: SkillSpec,
        inputs: dict[str, Any],
        *,
        json_mode: bool = True,
        temperature: float = 0.3,
        model: str | None = None,
        strict_inputs: bool = True,
        inject_long_term_memory: bool = True,
    ) -> SkillResult:
        """Render the SKILL with `inputs`, call the LLM, store the run, return SkillResult.

        When `json_mode=True` (the default for our scoring SKILLs), the runtime
        also tries to JSON-parse the reply; failure leaves `parsed=None` but
        the raw text is still returned and persisted for debugging.

        `strict_inputs=True` (the default) rejects keys that aren't declared in
        ``spec.inputs``. Set ``strict_inputs=False`` only when you explicitly
        want to pass debug context that is allowed to drift between calls.

        `inject_long_term_memory=True` (default) prepends relevant user_facts
        (mem0 v3-style retrieval) to the system message. Hash + persistence
        still see the *canonical* input, NOT the injected memory — same logical
        invocation hashes the same regardless of evolving memory state, so
        GEPA trainset deduplication stays correct. Set to False for SKILLs
        that explicitly don't want memory (rare).
        """
        canonical = _canonicalize_inputs(spec, inputs, strict=strict_inputs)

        system_msg = spec.body
        # ── Long-term memory injection (W12 fix: close the loop) ───────────
        # user_facts retrieve runs against a query built from the canonical
        # inputs (entities + key text), prepended to system_msg. Failure is
        # silent — memory is enrichment, not requirement.
        if inject_long_term_memory:
            try:
                from .. import user_facts as _uf
                memory_query = _build_memory_query(spec, canonical)
                memory_block = _uf.retrieve_for_prompt(
                    self._store, query=memory_query, top_k=8,
                )
                if memory_block:
                    system_msg = memory_block + "\n\n" + system_msg
            except Exception:
                pass

        user_msg = _render_inputs(spec, canonical)
        input_hash = _hash_invocation(spec, canonical)

        t0 = time.monotonic()
        resp = self._llm.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )
        latency_ms = resp.latency_ms or int((time.monotonic() - t0) * 1000)

        parsed: dict[str, Any] | None = None
        if json_mode:
            try:
                obj = json.loads(resp.content)
                parsed = obj if isinstance(obj, dict) else {"_value": obj}
            except json.JSONDecodeError:
                parsed = None

        run_id = self._record(
            spec=spec,
            input_hash=input_hash,
            inputs=canonical,
            output_text=resp.content,
            cost_usd=0.0,  # cost computation comes when we wire DeepSeek pricing in W4
            latency_ms=latency_ms,
        )

        return SkillResult(
            raw_text=resp.content,
            parsed=parsed,
            skill_name=spec.name,
            skill_version=spec.version,
            skill_run_id=run_id,
            input_hash=input_hash,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )

    def _record(
        self,
        *,
        spec: SkillSpec,
        input_hash: str,
        inputs: dict[str, Any],
        output_text: str,
        cost_usd: float,
        latency_ms: int,
    ) -> int:
        with self._store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "input_json, output_json, cost_usd, latency_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    spec.name,
                    spec.version,
                    input_hash,
                    json.dumps(inputs, ensure_ascii=False, default=str),
                    output_text,
                    cost_usd,
                    latency_ms,
                ),
            )
            return int(cur.lastrowid or 0)


def _canonicalize_inputs(
    spec: SkillSpec, inputs: dict[str, Any], *, strict: bool
) -> dict[str, Any]:
    """Return a dict containing exactly the spec-declared keys, in declared order.

    Rejects extra keys when ``strict`` is True. Always rejects missing required
    keys. When the spec declares no inputs, accepts the dict as-is (alphabetized
    for deterministic ordering downstream).
    """
    if not spec.inputs:
        return dict(sorted(inputs.items()))

    declared = list(spec.inputs)
    missing = [k for k in declared if k not in inputs]
    if missing:
        raise ValueError(
            f"SKILL {spec.name} requires inputs {declared}; missing: {missing}"
        )
    extra = [k for k in inputs if k not in declared]
    if extra and strict:
        raise ValueError(
            f"SKILL {spec.name} got unexpected inputs {extra}; declared: {declared}. "
            "Pass strict_inputs=False to suppress."
        )
    return {k: inputs[k] for k in declared}


def _render_inputs(spec: SkillSpec, inputs: dict[str, Any]) -> str:
    """Render canonical inputs as a labelled user-message section.

    Iterates ``inputs`` in dict order — by contract from `_canonicalize_inputs`,
    that order matches `spec.inputs` when declared, or alphabetical otherwise.
    """
    lines: list[str] = ["## 输入"]
    for key, val in inputs.items():
        lines.append(f"### {key}")
        lines.append(_stringify(val))
        lines.append("")
    if spec.output_schema:
        lines.append("## 输出 (严格 JSON，遵循以下 schema)")
        lines.append(spec.output_schema)
    else:
        lines.append("## 输出 (严格 JSON 对象)")
    return "\n".join(lines)


def _stringify(val: Any) -> str:
    if val is None:
        return "(空)"
    if isinstance(val, str):
        return val
    # pydantic v2 BaseModel
    dump = getattr(val, "model_dump_json", None)
    if callable(dump):
        return dump(indent=2)
    if isinstance(val, dict | list):
        return json.dumps(val, ensure_ascii=False, indent=2)
    return str(val)


def _build_memory_query(spec: SkillSpec, inputs: dict[str, Any]) -> str:
    """Construct a retrieve query from a SKILL's canonical inputs.

    We concatenate the SKILL name + the most-discriminating input values
    (company / role / job_text first lines / user_resume first lines).
    The query feeds user_facts.retrieve which uses Jaccard + entity match.

    Capped to 800 chars so the retrieval call is fast.
    """
    parts = [spec.name]
    # Prioritized keys — entities + short identifiers first
    for key in ("company", "role_focus", "role_hint", "role"):
        v = inputs.get(key)
        if v:
            parts.append(str(v))
    # Then short snippets from longer fields
    for key in ("job_text", "user_resume", "master_resume"):
        v = inputs.get(key)
        if v:
            parts.append(str(v)[:200])  # only first 200 chars
    return " ".join(parts)[:800]


def _hash_invocation(spec: SkillSpec, inputs: dict[str, Any]) -> str:
    """Stable hash over (skill name, version, canonical inputs).

    Inputs are stringified the same way `_render_inputs` does, so the hash key
    encodes exactly the prompt the LLM saw. Two invocations with byte-identical
    rendered prompts always hash equally; any change to a value (including a
    Pydantic model's field) flows through to a new hash.
    """
    payload = {
        "name": spec.name,
        "version": spec.version,
        "inputs": {k: _stringify(v) for k, v in inputs.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
