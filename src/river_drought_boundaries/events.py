"""Drought-event detection and one-to-one association.

The functions in this module implement the definitions used by the event
catalogue. Dates must be monthly and ordered without duplicate basin-months.
Missing index values split an event sequence and right-censor an open event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class EventDefinition:
    """Threshold and persistence settings for a monthly drought event."""

    onset_run: int = 3
    onset_threshold: float = -1.0
    termination_run: int = 2
    termination_threshold: float = -0.5

    def __post_init__(self) -> None:
        if self.onset_run < 1 or self.termination_run < 1:
            raise ValueError("Persistence lengths must be positive integers")
        if self.termination_threshold <= self.onset_threshold:
            raise ValueError("Termination threshold must exceed onset threshold")


def detect_events(
    values: np.ndarray,
    dates: np.ndarray,
    definition: EventDefinition = EventDefinition(),
) -> list[dict[str, object]]:
    """Detect non-overlapping drought events in one monthly index series.

    Onset is the first month of a complete run at or below the onset
    threshold. Termination is the last drought month before a complete run
    above the termination threshold. An event reaching a missing month or the
    record boundary is right-censored.
    """

    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates)
    if values.ndim != 1 or dates.ndim != 1 or len(values) != len(dates):
        raise ValueError("values and dates must be one-dimensional arrays of equal length")

    events: list[dict[str, object]] = []
    index = 0
    while index < len(values):
        onset_slice = values[index : index + definition.onset_run]
        if (
            len(onset_slice) == definition.onset_run
            and np.isfinite(onset_slice).all()
            and (onset_slice <= definition.onset_threshold).all()
        ):
            onset = index
            cursor = index + definition.onset_run
            censored_by_missing = False
            while cursor < len(values):
                if not np.isfinite(values[cursor]):
                    censored_by_missing = True
                    break
                recovery_slice = values[cursor : cursor + definition.termination_run]
                if (
                    len(recovery_slice) == definition.termination_run
                    and np.isfinite(recovery_slice).all()
                    and (recovery_slice > definition.termination_threshold).all()
                ):
                    break
                cursor += 1

            right_censored = cursor >= len(values) or censored_by_missing
            termination = cursor - 1
            event_values = values[onset : termination + 1]
            finite = event_values[np.isfinite(event_values)]
            events.append(
                {
                    "onset_idx": onset,
                    "term_idx": termination,
                    "onset": dates[onset],
                    "termination": dates[termination],
                    "right_censored": right_censored,
                    "censored_by_missing": censored_by_missing,
                    "duration_months": termination - onset + 1,
                    "severity": float(finite.sum()) if len(finite) else np.nan,
                }
            )
            index = termination + 1
            continue
        index += 1
    return events


def containing_event(events: list[dict[str, object]], month_index: int) -> int | None:
    """Return the position of the event containing a monthly index."""

    if not events:
        return None
    onsets = np.fromiter((int(event["onset_idx"]) for event in events), dtype=int)
    position = int(np.searchsorted(onsets, month_index, side="right") - 1)
    if position >= 0 and month_index <= int(events[position]["term_idx"]):
        return position
    return None


def admissible_edges(
    meteorological_events: list[dict[str, object]],
    river_events: list[dict[str, object]],
    initially_at_risk_indices: list[int],
    window_months: int,
) -> list[dict[str, int]]:
    """Construct admissible meteorological-to-river onset associations."""

    if window_months < 0:
        raise ValueError("window_months must be non-negative")
    edges: list[dict[str, int]] = []
    for meteorological_index in initially_at_risk_indices:
        meteorological = meteorological_events[meteorological_index]
        meteorological_onset = int(meteorological["onset_idx"])
        meteorological_termination = int(meteorological["term_idx"])
        for river_index, river in enumerate(river_events):
            river_onset = int(river["onset_idx"])
            lag = river_onset - meteorological_onset
            if lag < 0 or river_onset > meteorological_termination + window_months:
                continue
            overlap_start = max(meteorological_onset, river_onset)
            overlap_end = min(meteorological_termination, int(river["term_idx"]))
            overlap = max(0, overlap_end - overlap_start + 1)
            edges.append(
                {
                    "met_index": meteorological_index,
                    "hyd_index": river_index,
                    "lag": lag,
                    "overlap": overlap,
                }
            )
    return edges


def maximum_one_to_one(
    meteorological_indices: list[int],
    river_event_count: int,
    edges: list[dict[str, int]],
) -> list[dict[str, int]]:
    """Select maximum-cardinality one-to-one pairs with deterministic ties.

    The score first maximizes the number of pairs, then overlap in months, and
    finally favors the shorter non-negative onset lag.
    """

    if not meteorological_indices or river_event_count < 1 or not edges:
        return []
    row_lookup = {
        meteorological_index: row
        for row, meteorological_index in enumerate(meteorological_indices)
    }
    score = np.zeros((len(meteorological_indices), river_event_count), dtype=float)
    lookup: dict[tuple[int, int], dict[str, int]] = {}
    for edge in edges:
        row = row_lookup[edge["met_index"]]
        column = edge["hyd_index"]
        score[row, column] = 1_000_000_000 + 1_000 * edge["overlap"] - edge["lag"]
        lookup[(row, column)] = edge
    rows, columns = linear_sum_assignment(score, maximize=True)
    return [
        lookup[(row, column)]
        for row, column in zip(rows, columns, strict=True)
        if (row, column) in lookup
    ]
