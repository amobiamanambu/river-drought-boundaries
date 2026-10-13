from __future__ import annotations

import unittest

import numpy as np

from river_drought_boundaries.events import (
    EventDefinition,
    admissible_edges,
    containing_event,
    detect_events,
    maximum_one_to_one,
)


class EventDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = EventDefinition()

    def test_complete_event_uses_last_month_before_confirmed_recovery(self) -> None:
        values = np.array([-0.2, -1.1, -1.2, -1.3, -0.8, -0.4, -0.3])
        dates = np.arange(len(values))
        events = detect_events(values, dates, self.definition)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["onset_idx"], 1)
        self.assertEqual(events[0]["term_idx"], 4)
        self.assertEqual(events[0]["duration_months"], 4)
        self.assertFalse(events[0]["right_censored"])

    def test_missing_month_right_censors_open_event(self) -> None:
        values = np.array([-1.2, -1.1, -1.3, -0.8, np.nan, -0.2])
        dates = np.arange(len(values))
        event = detect_events(values, dates, self.definition)[0]
        self.assertEqual(event["term_idx"], 3)
        self.assertTrue(event["right_censored"])
        self.assertTrue(event["censored_by_missing"])

    def test_containing_event_respects_boundaries(self) -> None:
        events = [{"onset_idx": 2, "term_idx": 5}, {"onset_idx": 9, "term_idx": 11}]
        self.assertIsNone(containing_event(events, 1))
        self.assertEqual(containing_event(events, 2), 0)
        self.assertEqual(containing_event(events, 5), 0)
        self.assertIsNone(containing_event(events, 6))
        self.assertEqual(containing_event(events, 10), 1)


class AssociationTests(unittest.TestCase):
    def test_admissible_edges_require_nonnegative_lag_and_window(self) -> None:
        meteorological = [{"onset_idx": 3, "term_idx": 6}]
        river = [
            {"onset_idx": 2, "term_idx": 5},
            {"onset_idx": 4, "term_idx": 8},
            {"onset_idx": 13, "term_idx": 15},
        ]
        edges = admissible_edges(meteorological, river, [0], window_months=6)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["hyd_index"], 1)
        self.assertEqual(edges[0]["lag"], 1)

    def test_assignment_maximizes_cardinality_before_overlap(self) -> None:
        edges = [
            {"met_index": 0, "hyd_index": 0, "lag": 0, "overlap": 10},
            {"met_index": 0, "hyd_index": 1, "lag": 1, "overlap": 1},
            {"met_index": 1, "hyd_index": 0, "lag": 0, "overlap": 9},
        ]
        selected = maximum_one_to_one([0, 1], 2, edges)
        pairs = {(edge["met_index"], edge["hyd_index"]) for edge in selected}
        self.assertEqual(pairs, {(0, 1), (1, 0)})


if __name__ == "__main__":
    unittest.main()
