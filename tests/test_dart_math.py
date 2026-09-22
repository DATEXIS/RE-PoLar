"""Pure-function tests for re_polar/datasets/dart_math.py (no network)."""

import pytest

from re_polar.datasets.dart_math import extract_boxed_answer, parse_level


def test_boxed_simple():
    assert extract_boxed_answer(r"thus \boxed{42}.") == "42"


def test_boxed_nested_braces():
    assert extract_boxed_answer(r"so \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_boxed_takes_last():
    assert extract_boxed_answer(r"\boxed{1} wait, no: \boxed{2}") == "2"


def test_boxed_missing_returns_none():
    assert extract_boxed_answer("no answer here") is None
    assert extract_boxed_answer(r"broken \boxed{unclosed") is None


def test_parse_level():
    assert parse_level("Level 3") == 3
    assert parse_level("Level ?") is None
