"""Capture submitted settings separately from resolved defaults."""

from dataclasses import dataclass

from .job import JobSpec


@dataclass(frozen=True)
class JobConfigurations:
    """Immutable JSON snapshots of one validated request."""

    submitted_json: str
    resolved_json: str


def resolve_job(request: dict[str, object]) -> JobConfigurations:
    """Validate a request and capture both configuration forms."""

    validated = JobSpec.model_validate(request)

    return JobConfigurations(
        submitted_json=validated.model_dump_json(
            exclude_unset=True,
            indent=2,
        ),
        resolved_json=validated.model_dump_json(indent=2),
    )