"""Scheduling helpers for batched prefill/decode and mixed traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def group_request_ids_by_sequence_length(
    request_ids: Sequence[int],
    sequence_lengths: Sequence[int],
) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for request_id, sequence_length in zip(request_ids, sequence_lengths):
        grouped[int(sequence_length)].append(int(request_id))
    return dict(grouped)


def group_pairs_by_sequence_length(
    request_ids: Sequence[int],
    token_ids: Sequence[int],
    sequence_lengths: Sequence[int],
) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for request_id, token_id, sequence_length in zip(request_ids, token_ids, sequence_lengths):
        grouped[int(sequence_length)].append((int(request_id), int(token_id)))
    return dict(grouped)
