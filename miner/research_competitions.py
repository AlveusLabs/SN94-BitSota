from __future__ import annotations

from dataclasses import dataclass
import enum


class ParticipationStyle(str, enum.Enum):
    direct = "direct"
    pool = "pool"


class CompetitionMode(str, enum.Enum):
    standard = "standard"
    centerless = "centerless"
    peer_evaluation = "peer_evaluation"


class EvaluationMode(str, enum.Enum):
    validator = "validator"
    peer_consensus = "peer_consensus"


class MetricDirection(str, enum.Enum):
    maximize = "maximize"
    minimize = "minimize"


@dataclass(frozen=True, slots=True)
class ResearchCompetitionTemplate:
    slug: str
    title: str
    repository: str
    base_ref: str
    metric_name: str
    metric_direction: MetricDirection
    summary: str


BUILTIN_RESEARCH_COMPETITIONS: tuple[ResearchCompetitionTemplate, ...] = (
    ResearchCompetitionTemplate(
        slug="nanogpt-default",
        title="Default nanoGPT-style five-minute replay",
        repository="https://github.com/karpathy/autoresearch",
        base_ref="master",
        metric_name="val_bpb",
        metric_direction=MetricDirection.minimize,
        summary="Fixed five-minute nanoGPT replay with train.py-only changes.",
    ),
    ResearchCompetitionTemplate(
        slug="gpt2-145m-ternary-2x",
        title="GPT-2 145M ternary decomposition at 2x+ compression",
        repository="https://github.com/karpathy/autoresearch",
        base_ref="master",
        metric_name="heldout_ppl",
        metric_direction=MetricDirection.minimize,
        summary="Checkpoint decomposition with ternary and compression constraints.",
    ),
    ResearchCompetitionTemplate(
        slug="gpt2-145m-compression-2x",
        title="GPT-2 145M decomposition at 2x+ compression",
        repository="https://github.com/karpathy/autoresearch",
        base_ref="master",
        metric_name="heldout_ppl",
        metric_direction=MetricDirection.minimize,
        summary="Compression-only decomposition variant for GPT-2 145M.",
    ),
    ResearchCompetitionTemplate(
        slug="eggroll-efficiency",
        title="HyperscaleES Eggroll speed and sample-efficiency",
        repository="https://github.com/ESHyperscale/HyperscaleES",
        base_ref="main",
        metric_name="sample_efficiency_score",
        metric_direction=MetricDirection.maximize,
        summary="Evolution-strategy efficiency benchmark with fixed runtime and sample budgets.",
    ),
    ResearchCompetitionTemplate(
        slug="bitnet-cpu-ternary-kernel",
        title="Fastest CPU ternary kernel for a BitNet-style LLM",
        repository="https://github.com/microsoft/BitNet",
        base_ref="main",
        metric_name="tokens_per_second",
        metric_direction=MetricDirection.maximize,
        summary="CPU-only ternary kernel competition for a fixed small BitNet-style benchmark.",
    ),
)


def list_builtin_research_competitions() -> list[ResearchCompetitionTemplate]:
    return list(BUILTIN_RESEARCH_COMPETITIONS)


def get_builtin_research_competition(slug: str) -> ResearchCompetitionTemplate:
    wanted = str(slug or "").strip().lower()
    for template in BUILTIN_RESEARCH_COMPETITIONS:
        if template.slug == wanted:
            return template
    raise KeyError(slug)
