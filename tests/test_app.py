"""
Tests for the Gradio callbacks in app.py. The agent is mocked, so no API keys are needed.
"""

import csv

import pytest

import app

RESTOCK = {
    "decision": "restock",
    "confidence": 0.8,
    "issue_type": "low stock",
    "verdict_summary": "Reorder soon.",
    "key_findings": ["Only 6 cartons left"],
    "recommendations": ["Order 40 cartons"],
    "reorder_quantity": 40,
    "discount_percent": None,
    "reasoning": "Sells about 10 a day.",
    "standard_referenced": None,
}


def fake_result(decision_obj):
    return {
        "decision": decision_obj,
        "raw_text": "raw reply",
        "tool_log": [],
        "trace": ["perceive: got report", "act: decided"],
    }


@pytest.fixture(autouse=True)
def gemini_key_present(monkeypatch):
    monkeypatch.setattr(app.agent, "keys_configured", lambda: {"gemini": True, "tavily": True})


def run(monkeypatch, decision_obj, state=None, item="Milk 1L", report="6 cartons left"):
    monkeypatch.setattr(app.agent, "review_item", lambda *a, **k: fake_result(decision_obj))
    return app.run_review(item, report, state or app.new_state())


def test_restock_shows_reorder_quantity(monkeypatch):
    badge, reasoning, tool_html, trace, stats, log_update, state, dropdown = run(monkeypatch, RESTOCK)

    assert "RESTOCK" in badge
    assert "Suggested reorder: 40 units" in badge
    assert "**Reorder soon.**" in reasoning
    assert "Key findings" in reasoning and "Only 6 cartons left" in reasoning
    assert "Recommended next steps" in reasoning and "Order 40 cartons" in reasoning
    assert state["log"][0]["id"] == "G-1001"
    assert state["log"][0]["decision"] == "restock"
    assert "Reviewed" in stats
    assert dropdown["choices"] == ["G-1001"]
    assert len(log_update["value"]) == 1 and len(log_update["value"][0]) == 7


def test_discount_percent_accepts_strings(monkeypatch):
    obj = dict(RESTOCK, decision="discount", reorder_quantity=None, discount_percent="25%")
    badge, *_ = run(monkeypatch, obj)
    assert "DISCOUNT" in badge
    assert "Suggested markdown: 25%" in badge


@pytest.mark.parametrize("raw, expected", [("REMOVE", "remove"), ("accept", "escalate"), (None, "escalate")])
def test_decision_is_normalised_or_escalated(monkeypatch, raw, expected):
    *_, state, _ = run(monkeypatch, dict(RESTOCK, decision=raw))
    assert state["log"][0]["decision"] == expected


def test_unparseable_agent_reply_escalates(monkeypatch):
    badge, reasoning, *_ , state, _ = run(monkeypatch, None)
    assert "ESCALATE" in badge
    assert "raw reply" in reasoning
    assert state["log"][0]["decision"] == "escalate"


def test_validation_message_when_fields_are_empty():
    err, *_ = app.run_review("", "", app.new_state())
    assert "fill in both" in err


def test_missing_gemini_key_message(monkeypatch):
    monkeypatch.setattr(app.agent, "keys_configured", lambda: {"gemini": False, "tavily": True})
    _, message, *_ = app.run_review("Milk", "report", app.new_state())
    assert "GEMINI_API_KEY is not set" in message


def test_agent_errors_are_shown_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(app.agent, "review_item", boom)
    _, message, *_ = app.run_review("Milk", "report", app.new_state())
    assert "Agent error" in message and "quota exceeded" in message


def test_correction_updates_memory_and_log(monkeypatch):
    *_, state, _ = run(monkeypatch, RESTOCK)

    status, log_update, state, dropdown = app.submit_correction(
        "G-1001", "keep", "New delivery arrives tomorrow", state
    )

    assert "G-1001" in status
    assert state["log"][0]["human_review"] == "keep"
    assert "corrected to KEEP" in state["memory_notes"]
    assert "New delivery arrives tomorrow" in state["memory_notes"]
    assert log_update["value"][0][6] == "KEEP"
    assert dropdown["choices"] == []


def test_confirming_the_agent_is_not_called_a_correction(monkeypatch):
    *_, state, _ = run(monkeypatch, RESTOCK)
    _, _, state, _ = app.submit_correction("G-1001", "restock", "", state)
    assert "confirmed it" in state["memory_notes"]
    assert "corrected" not in state["memory_notes"]


def test_correction_requires_a_selection():
    html, *_ = app.submit_correction(None, "keep", "", app.new_state())
    assert "Select an item" in html


def test_stats_count_each_decision(monkeypatch):
    state = app.new_state()
    for decision in ("keep", "restock", "restock", "remove"):
        *_, state, _ = run(monkeypatch, dict(RESTOCK, decision=decision), state=state)
    html = app.stats_html(state)
    assert ">4</div>" in html  # reviewed
    assert ">2</div>" in html  # restock


def test_export_csv(monkeypatch):
    *_, state, _ = run(monkeypatch, RESTOCK)

    path = app.export_csv(state)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["item"] == "Milk 1L"
    assert rows[0]["decision"] == "restock"
    assert rows[0]["reorder_quantity"] == "40"
    assert rows[0]["discount_percent"] == ""
    assert rows[0]["recommendations"] == "Order 40 cartons"


def test_export_csv_with_empty_log_returns_none():
    assert app.export_csv(app.new_state()) is None


def test_reset_all_returns_a_clean_state():
    outputs = app.reset_all()
    assert len(outputs) == 8
    assert outputs[0] == {"log": [], "memory_notes": "", "seq": 0}
