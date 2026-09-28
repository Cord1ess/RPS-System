import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fail_on_errors_in_qt_callbacks(monkeypatch):
    """
    Qt prints an exception raised inside a signal handler and carries on, so a broken handler would
    not fail a test. Collect them through sys.excepthook and fail the test instead.
    """
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, tb: errors.append(value))
    yield
    assert not errors, f"exception(s) inside Qt callbacks: {errors!r}"
