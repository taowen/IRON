"""Q4NX projection phase chunk schedule shared by generators and references."""

from __future__ import annotations

from dataclasses import dataclass

from contract import PHASE_BLOCKS, PHASE_CHUNKS, ROWS_PER_COLUMN, ROWS_PER_PATCH

Q_BODY_RECORDS = PHASE_BLOCKS[0]
KV_BODY_RECORDS = PHASE_BLOCKS[1]
O_BODY_RECORDS = PHASE_BLOCKS[3]
DOWN_BODY_RECORDS = PHASE_BLOCKS[6]
UPGATE_REPLAYS = PHASE_BLOCKS[4] + PHASE_BLOCKS[5]
PATCHES_PER_COLUMN = ROWS_PER_COLUMN // ROWS_PER_PATCH


@dataclass(frozen=True)
class ProjectionPlan:
    label: str
    records: int
    chunks_per_record: int
    weight_chunk_base: int

    @property
    def weight_chunks(self) -> int:
        return self.records * self.chunks_per_record


def _plan_with_bases(items: tuple[tuple[str, int, int], ...]) -> tuple[ProjectionPlan, ...]:
    base = 0
    plans: list[ProjectionPlan] = []
    for label, records, chunks_per_record in items:
        plan = ProjectionPlan(label, records, chunks_per_record, base)
        plans.append(plan)
        base += plan.weight_chunks
    return tuple(plans)


QKV_BODY_PLANS = _plan_with_bases(
    (
        ("q", Q_BODY_RECORDS, PHASE_CHUNKS[0]),
        ("k", KV_BODY_RECORDS, PHASE_CHUNKS[1]),
        ("v", KV_BODY_RECORDS, PHASE_CHUNKS[2]),
    )
)

CURRENT_TAIL_PLANS = _plan_with_bases(
    (
        ("o", O_BODY_RECORDS, PHASE_CHUNKS[3]),
        ("upgate", UPGATE_REPLAYS, PHASE_CHUNKS[4]),
        ("down", DOWN_BODY_RECORDS, PHASE_CHUNKS[6]),
    )
)


def _require_plan(plans: tuple[ProjectionPlan, ...], label: str) -> ProjectionPlan:
    for plan in plans:
        if plan.label == label:
            return plan
    raise ValueError(f"unknown projection plan: {label}")


def current_tail_plan(label: str) -> ProjectionPlan:
    return _require_plan(CURRENT_TAIL_PLANS, label)


def qkv_body_plan(label: str) -> ProjectionPlan:
    return _require_plan(QKV_BODY_PLANS, label)


Q_WEIGHT_CHUNK_BASE = qkv_body_plan("q").weight_chunk_base
Q_CHUNKS_PER_RECORD = qkv_body_plan("q").chunks_per_record
Q_WEIGHT_CHUNKS = qkv_body_plan("q").weight_chunks
K_WEIGHT_CHUNK_BASE = qkv_body_plan("k").weight_chunk_base
K_CHUNKS_PER_RECORD = qkv_body_plan("k").chunks_per_record
K_WEIGHT_CHUNKS = qkv_body_plan("k").weight_chunks
V_WEIGHT_CHUNK_BASE = qkv_body_plan("v").weight_chunk_base
V_CHUNKS_PER_RECORD = qkv_body_plan("v").chunks_per_record
V_WEIGHT_CHUNKS = qkv_body_plan("v").weight_chunks
QKV_BODY_WEIGHT_CHUNKS = sum(plan.weight_chunks for plan in QKV_BODY_PLANS)

O_WEIGHT_CHUNK_BASE = current_tail_plan("o").weight_chunk_base
O_CHUNKS_PER_RECORD = current_tail_plan("o").chunks_per_record
O_WEIGHT_CHUNKS = current_tail_plan("o").weight_chunks
UPGATE_WEIGHT_CHUNK_BASE = current_tail_plan("upgate").weight_chunk_base
UPGATE_CHUNKS_PER_REPLAY = current_tail_plan("upgate").chunks_per_record
UPGATE_WEIGHT_CHUNKS = current_tail_plan("upgate").weight_chunks
DOWN_WEIGHT_CHUNK_BASE = current_tail_plan("down").weight_chunk_base
DOWN_CHUNKS = current_tail_plan("down").chunks_per_record
DOWN_WEIGHT_CHUNKS = current_tail_plan("down").weight_chunks
TOTAL_WEIGHT_CHUNKS = sum(plan.weight_chunks for plan in CURRENT_TAIL_PLANS)

FULL_LAYER_O_WEIGHT_CHUNK_BASE = QKV_BODY_WEIGHT_CHUNKS + O_WEIGHT_CHUNK_BASE
FULL_LAYER_UPGATE_WEIGHT_CHUNK_BASE = QKV_BODY_WEIGHT_CHUNKS + UPGATE_WEIGHT_CHUNK_BASE
FULL_LAYER_DOWN_WEIGHT_CHUNK_BASE = QKV_BODY_WEIGHT_CHUNKS + DOWN_WEIGHT_CHUNK_BASE
FULL_LAYER_TOTAL_WEIGHT_CHUNKS = QKV_BODY_WEIGHT_CHUNKS + TOTAL_WEIGHT_CHUNKS
