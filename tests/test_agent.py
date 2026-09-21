"""
Tests for agent.py. Gemini and Tavily are mocked, so no API keys or network are needed.
The mocked Gemini responses are built from the real google-genai types.
"""

import copy
import json
from datetime import date

import pytest
from google.genai import types

import agent

DECISION = {
    "decision": "remove",
    "confidence": 0.93,
    "issue_type": "near expiry",
    "verdict_summary": "This milk should come off the shelf today.",
    "key_findings": ["Use-by date was 2026-09-19"],
    "recommendations": ["Pull the batch"],
    "reorder_quantity": None,
    "discount_percent": None,
    "reasoning": "Past use-by date.",
    "standard_referenced": None,
}
FINAL_TEXT = "Here is my call.\n<decision>\n" + json.dumps(DECISION, indent=2) + "\n</decision>"


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": copy.deepcopy(contents), "config": config})
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def call_response(*queries):
    """A model turn that asks for one web search per query."""
    parts = [
        types.Part(
            function_call=types.FunctionCall(
                id=f"fc_{i}", name=agent.SEARCH_TOOL_NAME, args={"query": q}
            )
        )
        for i, q in enumerate(queries)
    ]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def text_response(text):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


@pytest.fixture
def install(monkeypatch):
    """Install a scripted fake Gemini client and a fake Tavily search."""
    monkeypatch.setattr(
        agent,
        "run_tavily_search",
        lambda query, max_results=4: [{"title": "T", "url": "https://example.org", "snippet": "s"}],
    )

    def _install(script):
        client = FakeClient(script)
        monkeypatch.setattr(agent, "get_gemini_client", lambda: client)
        return client

    return _install


# ---------------------------------------------------------------------------


def test_extract_decision_json_variants():
    assert agent.extract_decision_json(FINAL_TEXT)["decision"] == "remove"
    assert agent.extract_decision_json(json.dumps(DECISION))["confidence"] == 0.93
    assert agent.extract_decision_json("```json\n" + json.dumps(DECISION) + "\n```")["decision"] == "remove"
    assert agent.extract_decision_json("no json here") is None
    assert agent.extract_decision_json("") is None


def test_tool_declaration_is_valid():
    decl = agent.SEARCH_TOOL.function_declarations[0]
    assert decl.name == "search_grocery_reference"
    assert decl.parameters.required == ["query"]
    assert "query" in decl.parameters.properties


def test_search_round_then_decision(install):
    client = install([call_response("opened milk shelf life"), text_response(FINAL_TEXT)])

    out = agent.review_item(
        "Full-cream milk 1L",
        "6 cartons left, use-by 2026-09-19, sold 10 a day.",
        memory_notes="Managers prefer removing dairy on the use-by date.",
        today="2026-09-20",
    )

    assert out["decision"]["decision"] == "remove"
    assert out["tool_log"][0]["query"] == "opened milk shelf life"
    assert out["trace"][0].startswith("perceive:")
    assert any(t.startswith("reason:") for t in out["trace"])
    assert any("search_grocery_reference returned 1" in t for t in out["trace"])
    assert out["trace"][-1] == "act: decision = remove (confidence 0.93)"

    first, second = client.models.calls
    assert first["model"] == agent.GEMINI_MODEL

    user_text = first["contents"][0].parts[0].text
    assert "Today's date: 2026-09-20" in user_text
    assert "Item: Full-cream milk 1L" in user_text
    assert "Staff report:" in user_text

    assert "Managers prefer removing dairy" in first["config"].system_instruction
    assert first["config"].tools[0].function_declarations[0].name == agent.SEARCH_TOOL_NAME
    assert first["config"].automatic_function_calling.disable is True

    # The second request replays: user turn, the model's function call, our function response.
    assert [c.role for c in second["contents"]] == ["user", "model", "user"]
    assert second["contents"][1].parts[0].function_call.name == agent.SEARCH_TOOL_NAME
    fr = second["contents"][2].parts[0].function_response
    assert fr.id == "fc_0"
    assert fr.name == agent.SEARCH_TOOL_NAME
    assert fr.response["results"][0]["title"] == "T"


def test_parallel_calls_are_answered_in_one_turn(install):
    client = install([call_response("q1", "q2"), text_response(FINAL_TEXT)])

    out = agent.review_item("Milk", "some report")

    assert [e["query"] for e in out["tool_log"]] == ["q1", "q2"]
    last_turn = client.models.calls[1]["contents"][-1]
    assert [p.function_response.id for p in last_turn.parts] == ["fc_0", "fc_1"]


def test_direct_decision_uses_todays_date_by_default(install):
    client = install([text_response(FINAL_TEXT)])

    out = agent.review_item("Rice 5kg", "Plenty in stock, expiry next year.")

    assert f"Today's date: {date.today().isoformat()}" in client.models.calls[0]["contents"][0].parts[0].text
    assert out["tool_log"] == []
    assert out["decision"] is not None


def test_unparseable_reply(install):
    install([text_response("I think it is fine.")])

    out = agent.review_item("Rice 5kg", "Fine.")

    assert out["decision"] is None
    assert out["trace"][-1] == "act: agent replied but no structured decision could be parsed"


def test_round_limit(install):
    install([call_response("a"), call_response("b"), call_response("c")])

    out = agent.review_item("Rice 5kg", "Fine.", max_tool_rounds=2)

    assert out["decision"] is None
    assert len(out["tool_log"]) == 3
    assert out["trace"][-1] == "act: agent did not converge within the tool-call round limit"


def test_search_failure_is_reported_not_raised(install, monkeypatch):
    install([call_response("q"), text_response(FINAL_TEXT)])

    def boom(query, max_results=4):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "run_tavily_search", boom)

    out = agent.review_item("Milk", "some report")

    assert out["tool_log"][0]["results"][0]["title"] == "search failed"
    assert any("search failed" in t for t in out["trace"])
    assert out["decision"] is not None


def test_missing_gemini_key(monkeypatch):
    monkeypatch.setattr(agent, "GEMINI_API_KEY", "")
    monkeypatch.setattr(agent, "_gemini_client", None)

    assert agent.keys_configured()["gemini"] is False
    with pytest.raises(agent.MissingAPIKeyError, match="GEMINI_API_KEY"):
        agent.review_item("Rice 5kg", "Fine.")
