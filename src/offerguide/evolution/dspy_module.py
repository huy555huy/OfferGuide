"""Wrap a SkillSpec as a dspy.Module that GEPA can evolve.

Mapping:
- SkillSpec.body          → Signature.instructions   (the part GEPA evolves)
- SkillSpec.inputs        → InputField for each name (str-typed for now)
- SkillSpec.output_schema → single OutputField description (json string)

The output is a JSON string (`response_json`) rather than a structured Pydantic
field. We could let DSPy parse JSON for us, but that would couple GEPA's
candidate-rejection logic to our Pydantic schema's strictness — when a single
field is wrong, the whole prediction fails. Keeping the output as a free-form
JSON string lets the metric handle parsing errors as a soft signal (low score
+ explicit feedback string) rather than a hard rejection.
"""

from __future__ import annotations

from typing import Any

import dspy

from ..skills._spec import SkillSpec


def build_signature(spec: SkillSpec, *, instructions: str | None = None) -> Any:
    """Build a dspy.Signature class from a SKILL spec.

    `instructions` defaults to ``spec.body``; pass a different value to
    materialize a candidate prompt during GEPA's evolution loop.
    """
    if not spec.inputs:
        raise ValueError(
            f"SKILL {spec.name} has no declared inputs; can't build a DSPy signature."
        )

    fields: dict[str, tuple[type, Any]] = {}
    for name in spec.inputs:
        fields[name] = (str, dspy.InputField())
    fields["response_json"] = (
        str,
        dspy.OutputField(
            desc=(
                spec.output_schema
                or "Single JSON object matching the schema described in the system prompt."
            )
        ),
    )
    return dspy.Signature(fields, instructions=instructions or spec.body)


def build_student(spec: SkillSpec) -> dspy.Module:
    """Materialize a freshly-initialized student module for `spec`.

    Each call returns a new instance — GEPA needs to mutate the student's
    signature without bleed-through between runs.
    """
    sig = build_signature(spec)

    class _SkillStudent(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(sig)

        def forward(self, **inputs: Any) -> Any:
            return self.predict(**inputs)

    _SkillStudent.__name__ = f"{spec.name}_student"
    return _SkillStudent()


def read_instructions(student: dspy.Module) -> str:
    """Extract the current signature instructions from a student module.

    GEPA mutates ``student.predict.signature.instructions`` in place during
    optimization; after `optimizer.compile()` returns, this is where the
    evolved prompt lives.
    """
    return student.predict.signature.instructions  # type: ignore[no-any-return]
