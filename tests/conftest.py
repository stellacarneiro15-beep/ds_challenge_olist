"""Pytest reporting helpers for the challenge test suite."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

_GOALS_BY_NODEID: dict[str, str] = {}
_RESULTS: list[tuple[str, str, str]] = []

_STATUS_STYLES = {
    "PASSED": {"symbol": "✓", "label": "PASSED", "color": {"green": True}},
    "FAILED": {"symbol": "✗", "label": "FAILED", "color": {"red": True}},
    "SKIPPED": {"symbol": "-", "label": "SKIPPED", "color": {"yellow": True}},
}


def pytest_collection_modifyitems(items):
    """Store each test's docstring so the run can print a goal checklist."""
    _GOALS_BY_NODEID.clear()
    _RESULTS.clear()
    for item in items:
        _GOALS_BY_NODEID[item.nodeid] = inspect.getdoc(item.obj) or item.name


def pytest_runtest_logreport(report):
    """Remember each test result for the terminal summary."""
    if report.when != "call":
        return

    status = "PASSED" if report.passed else "FAILED" if report.failed else "SKIPPED"
    test_name = report.nodeid.split("::")[-1]
    goal = _GOALS_BY_NODEID.get(report.nodeid, test_name)
    _RESULTS.append((status, test_name, goal))


def pytest_terminal_summary(terminalreporter):
    """Print a clean checklist with each test, goal, and status."""
    if not _RESULTS:
        return

    terminalreporter.write_sep("-", "test goals")
    for i, (status, test_name, goal) in enumerate(_RESULTS, start=1):
        style = _STATUS_STYLES[status]
        line = f"{i}. {test_name} - {goal} - {style['symbol']} {style['label']}"
        terminalreporter.write_line(line, **style["color"])
