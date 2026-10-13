"""Reusable components for the river-drought event-boundary analysis."""

from .events import EventDefinition, admissible_edges, containing_event, detect_events, maximum_one_to_one

__all__ = [
    "EventDefinition",
    "admissible_edges",
    "containing_event",
    "detect_events",
    "maximum_one_to_one",
]

__version__ = "1.0.0"
