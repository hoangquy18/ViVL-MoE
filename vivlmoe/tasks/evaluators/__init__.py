"""Evaluator protocol and ordered gather helpers; concrete evaluators live in sibling modules."""

from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered, gather_ordered_tensors

__all__ = ["Evaluator", "gather_ordered", "gather_ordered_tensors"]
