"""
agent.py — Reasoning core of the Grocery Management Agent (Gemini edition).

This module implements the "Perceive -> Reason -> Tool-Use -> Act" cycle
using:

  - Google Gemini (google-genai SDK)  as the LLM reasoning / decision-making core
  - Tavily                            as a web-search tool the agent can call
                                      mid-reasoning, e.g. to look up shelf-life or
                                      storage guidance, a food-safety guideline,
                                      or a product recall notice

The agent is given an item name and a plain-language description of its
current situation (stock level, expiry date, storage, recent sales, anything
staff noticed). It decides, on its own, whether it needs more information
(and if so, calls the search tool), then returns a structured decision:
keep / restock / discount / remove / escalate, with a confidence score and
an explanation.
"""

import os
import re
import json
from datetime import date

from google import genai
from google.genai import types
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

# Optional. Left unset by default so the model's own default temperature is used.
_temp_raw = os.getenv("GEMINI_TEMPERATURE", "").strip()
try:
    GEMINI_TEMPERATURE = float(_temp_raw) if _temp_raw else None
except ValueError:
    GEMINI_TEMPERATURE = None

_gemini_client = None
_tavily_client = None


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key has not been configured."""


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise MissingAPIKeyError(
                "TAVILY_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def keys_configured() -> dict:
    return {"gemini": bool(GEMINI_API_KEY), "tavily": bool(TAVILY_API_KEY)}


# ---------------------------------------------------------------------------
# Tool definition (given to Gemini as a callable function)
# ---------------------------------------------------------------------------

SEARCH_TOOL_NAME = "search_grocery_reference"

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=SEARCH_TOOL_NAME,
            description=(
                "Search the web for food-safety guidance, shelf-life and storage "
                "information, product recall notices, seasonal demand or price "
                "trends, or suitable substitutes relevant to the grocery item being "
                "reviewed. Use this whenever you are unsure how long an item stays "
                "safe or sellable, when a recognised guideline (e.g. FSSAI, FDA, "
                "USDA, Codex) would make your decision more defensible, or when the "
                "item may be affected by a recall or shortage. Do not use it for "
                "routine situations, such as a well-stocked shelf-stable item far "
                "from its expiry date, or for items that are obviously unsafe, such "
                "as visibly mouldy produce."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="A focused, specific web search query.",
                    )
                },
                required=["query"],
            ),
        )
    ]
)


def run_tavily_search(query: str, max_results: int = 4) -> list:
    client = get_tavily_client()
    resp = client.search(query=query, max_results=max_results, search_depth="basic")
    results = resp.get("results", []) if isinstance(resp, dict) else []
    formatted = []
    for r in results:
        formatted.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or "")[:400],
            }
        )
    return formatted


# ---------------------------------------------------------------------------
# System prompt — defines the agent's role, tone, and required output shape
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the reasoning core of a Grocery Management Agent supporting the \
day-to-day running of a grocery store. A staff member gives you the name of an item and a \
plain-language description of its current situation: stock level, expiry or best-before date, \
storage conditions, recent sales, and anything they noticed. Your job is to decide what should \
happen to that item and explain it clearly in friendly, professional language.

Today's date is given at the top of each request. Use it to judge how close an item is to its \
expiry or best-before date.

You have one tool available: search_grocery_reference. Use it when you are unsure how long the \
item stays safe or sellable, when a recognised guideline (FSSAI, FDA, USDA, Codex, etc.) would \
help justify your decision, or when the item may be affected by a recall, shortage or seasonal \
change.

Decision rules:
- KEEP: no action is needed now; stock level, freshness and storage are all fine.
- RESTOCK: stock is low, or will run out soon at the current rate of sales; the item should be reordered.
- DISCOUNT: the item is still safe to sell but is close to its expiry or best-before date, \
overstocked, or slow-moving, so a markdown or promotion would reduce waste.
- REMOVE: the item is expired, spoiled, damaged, recalled or otherwise unsafe and must be pulled from sale.
- ESCALATE: you cannot confidently decide either way — a store manager should make the call.

Safety comes first. If there is any credible sign that an item may be unsafe to eat (spoilage, a \
broken cold chain, a recall, tampered or damaged packaging, or an expiry date that has passed), \
choose REMOVE or ESCALATE — never KEEP, RESTOCK or DISCOUNT.

If more than one action applies (for example an item that is both low in stock and close to \
expiry), choose the most urgent one as your decision and mention the others in the recommendations.

If key information is missing (for example no expiry date for a perishable item), say so in \
key_findings and lower your confidence. Choose ESCALATE if the missing information stops you \
from deciding safely.

When you have reached a final decision (do NOT call the tool again), end your reply with \
exactly one JSON object wrapped like this:

<decision>
{
  "decision": "keep" | "restock" | "discount" | "remove" | "escalate",
  "confidence": <number between 0 and 1>,
  "issue_type": "<short label, e.g. 'low stock', 'near expiry', 'spoilage', 'recall', 'overstock', or 'none'>",
  "verdict_summary": "<one friendly sentence summarising the outcome in plain language>",
  "key_findings": [
    "<specific observation 1 from the staff report>",
    "<specific observation 2 (or guideline / shelf-life threshold found)>"
  ],
  "recommendations": [
    "<actionable next step 1 in plain English>",
    "<actionable next step 2 if applicable>"
  ],
  "reorder_quantity": <whole number or null>,
  "discount_percent": <number or null>,
  "reasoning": "<2-3 sentence explanation for a store manager>",
  "standard_referenced": "<guideline/source name, or null>"
}
</decision>

Rules for the fields:
- verdict_summary: write for a non-expert. Example: "This milk should come off the shelf today — it passed its use-by date yesterday."
- key_findings: 1-3 bullet-length observations that drove the decision. Be specific (dates, quantities, thresholds).
- recommendations: 1-3 plain-English actions the staff member or store manager should take next.
- reorder_quantity: only for RESTOCK, and only if the report gives enough stock and sales information to justify a number. Otherwise null.
- discount_percent: only for DISCOUNT, as a suggested markdown percentage. Otherwise null.
- Never invent numbers you cannot justify from the report.
- The JSON must be valid and must be the last thing in your reply."""


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------

def extract_decision_json(text: str):
    if not text:
        return None
    m = re.search(r"<decision>(.*?)</decision>", text, re.S)
    raw = m.group(1).strip() if m else text.strip()
    raw = raw.strip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m2 = re.search(r"\{.*\}", raw, re.S)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main agent cycle: Perceive -> Reason -> Tool-Use -> Act
# ---------------------------------------------------------------------------

def _build_config(system_instruction: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[SEARCH_TOOL],
        temperature=GEMINI_TEMPERATURE,
        # We run the tool loop ourselves so every search shows up in the trace.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def review_item(
    item_name: str,
    situation_description: str,
    memory_notes: str = "",
    max_tool_rounds: int = 2,
    today: str = "",
) -> dict:
    """
    Runs one full review cycle for a grocery item.

    `today` is an ISO date string (e.g. "2026-09-20"). If left empty, the
    current date is used, so the agent can judge expiry dates.

    Returns a dict with:
      decision   -> parsed JSON dict (or None if parsing failed)
      raw_text   -> full text of the agent's final reply
      tool_log   -> list of {query, results} for every search call made
      trace      -> list of short human-readable strings describing each step,
                    useful for displaying an agent "trace" in the UI
    """
    client = get_gemini_client()

    system_instruction = SYSTEM_PROMPT
    if memory_notes.strip():
        system_instruction += (
            "\n\nLessons learned from prior corrections by store staff and managers. "
            "Apply these when relevant:\n" + memory_notes.strip()
        )
    config = _build_config(system_instruction)

    today_str = today.strip() or date.today().isoformat()
    user_msg = (
        f"Today's date: {today_str}\n"
        f"Item: {item_name}\n\n"
        f"Staff report:\n{situation_description}"
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]

    tool_log = []
    trace = ["perceive: received staff report for '%s'" % item_name]

    for round_num in range(max_tool_rounds + 1):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        function_calls = response.function_calls or []
        if function_calls:
            # Keep the model's own turn (including any thought signatures) in the history.
            contents.append(response.candidates[0].content)

            response_parts = []
            for fc in function_calls:
                args = dict(fc.args or {})
                query = str(args.get("query", ""))
                trace.append("reason: agent requested a web search — \"%s\"" % query)
                try:
                    results = run_tavily_search(query)
                    trace.append(
                        "tool-use: %s returned %d result(s)" % (SEARCH_TOOL_NAME, len(results))
                    )
                except MissingAPIKeyError as e:
                    results = [{"title": "search unavailable", "url": "", "snippet": str(e)}]
                    trace.append("tool-use: search failed — %s" % e)
                except Exception as e:
                    results = [{"title": "search failed", "url": "", "snippet": str(e)}]
                    trace.append("tool-use: search failed — %s" % e)

                tool_log.append({"query": query, "results": results})
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response={"results": results},
                        )
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))
            continue
        else:
            final_text = response.text or ""
            decision_obj = extract_decision_json(final_text)
            if decision_obj:
                trace.append(
                    "act: decision = %s (confidence %.2f)"
                    % (decision_obj.get("decision", "?"), float(decision_obj.get("confidence", 0) or 0))
                )
            else:
                trace.append("act: agent replied but no structured decision could be parsed")
            return {
                "decision": decision_obj,
                "raw_text": final_text,
                "tool_log": tool_log,
                "trace": trace,
            }

    trace.append("act: agent did not converge within the tool-call round limit")
    return {
        "decision": None,
        "raw_text": "The agent used too many tool calls without reaching a final decision.",
        "tool_log": tool_log,
        "trace": trace,
    }
