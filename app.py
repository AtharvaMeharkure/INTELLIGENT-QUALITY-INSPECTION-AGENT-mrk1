"""
app.py — Premium Gradio front-end for the Grocery Management Agent.

Run with:
    python app.py

Requires GEMINI_API_KEY and TAVILY_API_KEY to be set (see .env.example).
"""

import csv
import io
import os
import tempfile
from datetime import datetime, date as date_type

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
import agent  # noqa: E402  (import after load_dotenv so keys are picked up)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

VALID_DECISIONS = ("keep", "restock", "discount", "remove", "escalate")


def new_state():
    return {"log": [], "memory_notes": "", "seq": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers for model-supplied values
# ─────────────────────────────────────────────────────────────────────────────

def _as_number(value):
    """Best-effort conversion of a model-supplied value to a number (or None)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _as_list(value):
    """Turn a model-supplied value into a clean list of non-empty strings."""
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


def _bullets(items):
    """Markdown line-break bullets (renders inside a single paragraph)."""
    return "  \n".join(f"• {i}" for i in items)


def action_detail(decision, reorder_qty, discount_pct):
    """Short extra line shown under the decision badge (restock / discount only)."""
    if decision == "restock" and reorder_qty is not None:
        return f"Suggested reorder: {reorder_qty:g} units"
    if decision == "discount" and discount_pct is not None:
        return f"Suggested markdown: {discount_pct:g}%"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Date / freshness helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(val):
    """Parse a date value from Gradio (string 'YYYY-MM-DD' or None) → date or None."""
    if not val:
        return None
    if isinstance(val, date_type):
        return val
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def compute_date_context(mfg_date_val, expiry_date_val):
    """
    Given raw Gradio date values, return a human-readable context string that
    will be prepended to the staff report, plus a dict of computed metrics.
    """
    today = date_type.today()
    mfg   = _parse_date(mfg_date_val)
    exp   = _parse_date(expiry_date_val)

    lines = []
    metrics = {"mfg": mfg, "exp": exp, "today": today,
               "days_remaining": None, "total_shelf_days": None, "pct_used": None}

    if mfg:
        age_days = (today - mfg).days
        lines.append(f"Manufacturing date: {mfg.isoformat()} ({age_days} days ago)")
        metrics["age_days"] = age_days

    if exp:
        days_rem = (exp - today).days
        metrics["days_remaining"] = days_rem
        if days_rem < 0:
            lines.append(f"Expiry date: {exp.isoformat()} — EXPIRED {abs(days_rem)} day(s) ago")
        elif days_rem == 0:
            lines.append(f"Expiry date: {exp.isoformat()} — EXPIRES TODAY")
        else:
            lines.append(f"Expiry date: {exp.isoformat()} ({days_rem} day(s) remaining)")

        if mfg and exp > mfg:
            total = (exp - mfg).days
            used  = (today - mfg).days
            pct   = min(max(used / total * 100, 0), 100)
            metrics["total_shelf_days"] = total
            metrics["pct_used"] = pct
            lines.append(f"Shelf life: {used}/{total} days used ({pct:.0f}%)")

    return "\n".join(lines), metrics


def freshness_gauge_html(mfg_date_val, expiry_date_val):
    """Render a visual freshness gauge card."""
    _, m = compute_date_context(mfg_date_val, expiry_date_val)

    today = m["today"]
    mfg   = m.get("mfg")
    exp   = m.get("exp")
    pct   = m.get("pct_used")
    days_rem = m.get("days_remaining")

    if not mfg and not exp:
        return """
<div style="
    background:rgba(12,21,37,0.6); border:1px solid rgba(99,179,237,0.1);
    border-radius:14px; padding:18px 20px; margin-top:4px;
    font-family:'Inter',sans-serif; font-size:13px; color:#475569;
    font-style:italic;
">📅 Enter manufacturing and/or expiry dates above to see the freshness gauge.</div>"""

    # Colour based on state
    if exp and (exp - today).days < 0:
        bar_color = "#EF4444"; status_text = "EXPIRED"; status_color = "#F87171"
    elif exp and (exp - today).days == 0:
        bar_color = "#EF4444"; status_text = "EXPIRES TODAY"; status_color = "#FBBF24"
    elif pct is not None and pct >= 90:
        bar_color = "#EF4444"; status_text = "CRITICAL"; status_color = "#F87171"
    elif pct is not None and pct >= 75:
        bar_color = "#F59E0B"; status_text = "NEAR EXPIRY"; status_color = "#FBBF24"
    elif pct is not None and pct >= 50:
        bar_color = "#A78BFA"; status_text = "MONITOR"; status_color = "#A78BFA"
    else:
        bar_color = "#10B981"; status_text = "FRESH"; status_color = "#34D399"

    bar_pct = round(pct if pct is not None else 0, 1)

    # Build info chips
    chips = ""
    if mfg:
        chips += f'<span style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.2);color:#38BDF8;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;">Mfg: {mfg.isoformat()}</span>'
    if exp:
        chips += f'<span style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);color:#F87171;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-left:8px;">Exp: {exp.isoformat()}</span>'
    if days_rem is not None and days_rem >= 0:
        chips += f'<span style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.2);color:#A78BFA;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-left:8px;">{days_rem}d remaining</span>'

    bar_html = ""
    if pct is not None:
        bar_html = f"""
<div style="margin-top:14px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="font-size:11px;color:#64748B;font-weight:600;letter-spacing:0.5px;">SHELF LIFE CONSUMED</span>
    <span style="font-size:12px;color:{bar_color};font-weight:700;">{bar_pct:.0f}%</span>
  </div>
  <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;">
    <div style="height:100%;width:{bar_pct}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:999px;transition:width 0.4s ease;"></div>
  </div>
</div>"""

    return f"""
<div style="
    background:rgba(12,21,37,0.8); border:1px solid rgba(99,179,237,0.12);
    border-radius:14px; padding:18px 20px; margin-top:4px;
    font-family:'Inter',sans-serif;
">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <span style="font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748B;">📊 Freshness Status</span>
    <span style="font-size:11px;font-weight:800;letter-spacing:2px;color:{status_color};background:rgba(255,255,255,0.04);padding:4px 12px;border-radius:20px;border:1px solid {status_color}44;">{status_text}</span>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>
  {bar_html}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML component builders
# ─────────────────────────────────────────────────────────────────────────────

DECISION_STYLES = {
    "keep":     {"bg": "#011F13", "border": "#10B981", "text": "#34D399", "icon": "✓"},
    "restock":  {"bg": "#001A1F", "border": "#06B6D4", "text": "#22D3EE", "icon": "↻"},
    "discount": {"bg": "#150B2E", "border": "#8B5CF6", "text": "#A78BFA", "icon": "%"},
    "remove":   {"bg": "#1F0101", "border": "#EF4444", "text": "#F87171", "icon": "✗"},
    "escalate": {"bg": "#1F1000", "border": "#F59E0B", "text": "#FBBF24", "icon": "⚠"},
    "unknown":  {"bg": "#0A1120", "border": "#475569", "text": "#94A3B8", "icon": "—"},
}


def decision_badge_html(decision, confidence, detail=""):
    s = DECISION_STYLES.get(decision or "unknown", DECISION_STYLES["unknown"])
    label = (decision or "AWAITING").upper()
    conf_txt = (
        f"{confidence * 100:.0f}% confidence"
        if isinstance(confidence, (int, float))
        else "Run a review to get a decision"
    )
    detail_html = (
        f"""<div style="
        display:inline-block; margin-top:10px; padding:5px 12px;
        font-family:'Inter',sans-serif; font-size:12px; font-weight:600;
        color:{s['text']}; background:{s['border']}22;
        border:1px solid {s['border']}55; border-radius:8px;
    ">{detail}</div>"""
        if detail
        else ""
    )
    return f"""
<div style="
    display:flex; align-items:center; gap:18px;
    padding:22px 26px;
    background:{s['bg']};
    border:1.5px solid {s['border']};
    border-radius:14px;
    margin:8px 0 20px 0;
">
  <div style="
      width:54px; height:54px;
      background:{s['border']}1A;
      border:2px solid {s['border']};
      border-radius:50%;
      display:flex; align-items:center; justify-content:center;
      font-size:26px; color:{s['text']}; font-weight:800; flex-shrink:0;
  ">{s['icon']}</div>
  <div>
    <div style="
        font-family:'Inter',sans-serif; font-weight:800; font-size:26px;
        color:{s['text']}; letter-spacing:3px; line-height:1;
    ">{label}</div>
    <div style="
        font-family:'Inter',sans-serif; font-size:13px;
        color:{s['text']}88; margin-top:6px;
    ">{conf_txt}</div>
    {detail_html}
  </div>
</div>"""


def stats_html(state):
    log = state.get("log", [])
    total    = len(log)
    counts   = {d: sum(1 for r in log if r["decision"] == d) for d in VALID_DECISIONS}
    confs    = [r["confidence"] for r in log if isinstance(r["confidence"], (int, float))]
    avg_conf = f"{sum(confs) / len(confs) * 100:.0f}%" if confs else "—"

    def card(icon, val, label, color):
        return f"""
<div style="
    flex:1; min-width:110px;
    background:rgba(15,23,42,0.85);
    border:1px solid rgba(99,179,237,0.1);
    border-radius:14px; padding:18px 14px; text-align:center;
">
  <div style="font-size:22px; margin-bottom:8px;">{icon}</div>
  <div style="
      font-family:'Inter',sans-serif; font-size:28px; font-weight:800;
      color:{color}; line-height:1;
  ">{val}</div>
  <div style="
      font-family:'Inter',sans-serif; font-size:10px; font-weight:700;
      letter-spacing:1.2px; text-transform:uppercase; color:#475569; margin-top:6px;
  ">{label}</div>
</div>"""

    keep_c     = DECISION_STYLES["keep"]["border"]
    restock_c  = DECISION_STYLES["restock"]["border"]
    discount_c = DECISION_STYLES["discount"]["border"]
    remove_c   = DECISION_STYLES["remove"]["border"]
    escalate_c = DECISION_STYLES["escalate"]["border"]

    return f"""
<div style="display:flex; gap:12px; margin:0 0 24px 0; flex-wrap:wrap;">
  {card("📦", str(total),               "Reviewed",       "#38BDF8")}
  {card("✅", str(counts["keep"]),      "Keep",           keep_c)}
  {card("🛒", str(counts["restock"]),   "Restock",        restock_c)}
  {card("🏷️", str(counts["discount"]), "Discount",       discount_c)}
  {card("🗑️", str(counts["remove"]),   "Remove",         remove_c)}
  {card("⚠️", str(counts["escalate"]), "Escalated",      escalate_c)}
  {card("📊", avg_conf,                 "Avg Confidence", "#A78BFA")}
</div>"""


def trace_html(trace_lines):
    if not trace_lines:
        return """<div style="
            font-family:'JetBrains Mono',monospace; font-size:12px; color:#334155;
            padding:16px; background:rgba(15,23,42,0.5);
            border:1px solid rgba(99,179,237,0.07); border-radius:10px; font-style:italic;
        ">No trace yet. Run a review to see the agent reasoning steps.</div>"""

    step_colors = {
        "perceive": "#38BDF8",
        "reason":   "#A78BFA",
        "tool-use": "#F59E0B",
        "act":      "#10B981",
    }
    rows = ""
    for line in trace_lines:
        prefix = line.split(":")[0] if ":" in line else "info"
        color  = step_colors.get(prefix, "#64748B")
        rest   = line[len(prefix) + 1:].strip() if ":" in line else line
        rows  += f"""
<div style="display:flex; align-items:flex-start; gap:10px; padding:7px 0;
    border-bottom:1px solid rgba(99,179,237,0.04);">
  <span style="
      font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700;
      color:{color}; background:{color}18; padding:3px 8px; border-radius:4px;
      min-width:78px; text-align:center; text-transform:uppercase; letter-spacing:0.5px;
      margin-top:2px; flex-shrink:0;
  ">{prefix}</span>
  <span style="
      font-family:'JetBrains Mono',monospace; font-size:12px;
      color:#94A3B8; line-height:1.6;
  ">{rest}</span>
</div>"""

    return f"""<div style="
        background:rgba(8,11,20,0.7); border:1px solid rgba(99,179,237,0.1);
        border-radius:12px; padding:14px 16px;
        max-height:220px; overflow-y:auto;
    ">{rows}</div>"""


def tool_log_html(tool_log):
    if not tool_log:
        return """<div style="
            font-family:'Inter',sans-serif; font-size:13px; color:#334155;
            padding:16px; background:rgba(15,23,42,0.5);
            border:1px solid rgba(99,179,237,0.07); border-radius:10px; font-style:italic;
        ">No web searches were made for this item.</div>"""

    html = ""
    for entry in tool_log:
        results_html = ""
        for r in entry["results"]:
            title   = r.get("title")   or "(untitled)"
            url     = r.get("url")     or ""
            snippet = r.get("snippet") or ""
            link = (
                f'<a href="{url}" target="_blank" style="color:#38BDF8;text-decoration:none;'
                f'font-weight:600;font-size:13px;">{title}</a>'
                if url else
                f'<span style="color:#CBD5E1;font-weight:600;font-size:13px;">{title}</span>'
            )
            results_html += f"""
<div style="padding:10px 12px; margin:5px 0; background:rgba(15,23,42,0.8);
    border-radius:8px; border-left:3px solid rgba(56,189,248,0.35);">
  <div style="margin-bottom:4px;">{link}</div>
  <div style="font-family:'Inter',sans-serif; font-size:12px; color:#64748B; line-height:1.55;">{snippet}</div>
</div>"""

        html += f"""
<div style="margin-bottom:18px;">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
    <span style="font-size:14px;">🔍</span>
    <code style="
        font-family:'JetBrains Mono',monospace; font-size:11px; color:#F59E0B;
        background:rgba(245,158,11,0.1); padding:4px 10px; border-radius:6px;
        border:1px solid rgba(245,158,11,0.2);
    ">{entry['query']}</code>
  </div>
  {results_html}
</div>"""

    return f"""<div style="
        background:rgba(8,11,20,0.6); border:1px solid rgba(99,179,237,0.1);
        border-radius:12px; padding:16px; max-height:300px; overflow-y:auto;
    ">{html}</div>"""


def section_header(icon, title):
    return f"""<div style="
        display:flex; align-items:center; gap:10px;
        margin:24px 0 14px 0; padding-bottom:10px;
        border-bottom:1px solid rgba(99,179,237,0.1);
    ">
  <span style="font-size:16px;">{icon}</span>
  <span style="
      font-family:'Inter',sans-serif; font-size:11px; font-weight:700;
      letter-spacing:1.4px; text-transform:uppercase; color:#64748B;
  ">{title}</span>
</div>"""


HERO_HTML = """
<div style="
    background:linear-gradient(135deg,#0D1424 0%,#0F172A 60%,#0D1424 100%);
    border:1px solid rgba(99,179,237,0.15); border-radius:20px;
    padding:36px 40px; margin-bottom:4px; position:relative; overflow:hidden;
">
  <div style="
      position:absolute; top:0; left:0; right:0; bottom:0; pointer-events:none;
      background:
        radial-gradient(ellipse at 15% 50%, rgba(56,189,248,0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 50%, rgba(139,92,246,0.07) 0%, transparent 55%);
  "></div>
  <div style="position:relative; z-index:1;">
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
      <div style="
          width:52px; height:52px; flex-shrink:0;
          background:linear-gradient(135deg,#0284C7,#7C3AED);
          border-radius:14px; display:flex; align-items:center;
          justify-content:center; font-size:24px;
          box-shadow:0 4px 20px rgba(124,58,237,0.35);
      ">🛒</div>
      <div>
        <h1 style="
            font-family:'Inter',sans-serif; font-size:28px; font-weight:800; margin:0;
            background:linear-gradient(135deg,#38BDF8 0%,#A78BFA 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">Grocery Management Agent</h1>
        <p style="
            font-family:'Inter',sans-serif; font-size:13px; color:#64748B;
            margin:5px 0 0 0; letter-spacing:0.3px;
        ">Agentic AI &amp; Automation &nbsp;|&nbsp; Perceive → Reason → Tool-Use → Act → Reflect</p>
      </div>
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
      <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#38BDF8;background:rgba(56,189,248,0.1);padding:5px 13px;border-radius:20px;border:1px solid rgba(56,189,248,0.25);">🧠 Gemini LLM Reasoning</span>
      <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#F59E0B;background:rgba(245,158,11,0.1);padding:5px 13px;border-radius:20px;border:1px solid rgba(245,158,11,0.25);">🔎 Tavily Web Search</span>
      <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#10B981;background:rgba(16,185,129,0.1);padding:5px 13px;border-radius:20px;border:1px solid rgba(16,185,129,0.25);">👤 Human-in-the-Loop</span>
      <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#A78BFA;background:rgba(167,139,250,0.1);padding:5px 13px;border-radius:20px;border:1px solid rgba(167,139,250,0.25);">🔄 Memory &amp; Feedback</span>
    </div>
  </div>
</div>
"""

FOOTER_HTML = """
<div style="
    margin-top:32px; padding:20px 24px;
    background:rgba(15,23,42,0.5); border:1px solid rgba(99,179,237,0.08);
    border-radius:14px;
">
  <p style="
      font-family:'Inter',sans-serif; font-size:12px; color:#475569;
      line-height:1.8; margin:0;
  ">
    <strong style="color:#64748B;">How it works:</strong>
    Each run is one full agent cycle —
    <span style="color:#38BDF8;">perceive</span> (the staff report) →
    <span style="color:#A78BFA;">reason</span> (LLM weighs stock, freshness and food safety) →
    <span style="color:#F59E0B;">tool-use</span> (optionally searches the web for shelf-life, food-safety guidelines or recalls) →
    <span style="color:#10B981;">act</span> (structured keep / restock / discount / remove / escalate decision) →
    <span style="color:#EC4899;">reflect</span> (staff corrections are added to agent memory for future runs).
  </p>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_log_rows(state):
    return [
        [
            r["id"],
            r["time"],
            (r["item"][:28] + "…" if len(r["item"]) > 28 else r["item"]),
            r["issue_type"],
            r["decision"].upper(),
            f"{r['confidence']:.0%}" if isinstance(r["confidence"], (int, float)) else "—",
            r["human_review"].upper() if r["human_review"] else "⏳ pending",
        ]
        for r in reversed(state["log"])
    ]


def run_review(item_name, description, mfg_date_val, expiry_date_val, state):
    state = state or new_state()

    if not item_name.strip() or not description.strip():
        err = """<div style="
            font-family:'Inter',sans-serif; color:#F87171; font-size:14px;
            background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
            border-radius:10px; padding:14px 18px;
        ">⚠ Please fill in both the Item Name and the Staff Report.</div>"""
        return (err, "", tool_log_html([]), trace_html([]),
                stats_html(state), gr.update(value=[]), state, gr.update(choices=[], value=None))

    # Prepend date context if dates were provided
    date_context, _ = compute_date_context(mfg_date_val, expiry_date_val)
    full_description = description.strip()
    if date_context:
        full_description = date_context + "\n\n" + full_description

    keys = agent.keys_configured()
    if not keys["gemini"]:
        return (
            decision_badge_html("unknown", None),
            "**GEMINI_API_KEY is not set.** Check your `.env` file and restart the app.",
            tool_log_html([]), trace_html([]),
            stats_html(state), gr.update(value=[]), state, gr.update(),
        )

    try:
        result = agent.review_item(item_name, full_description, memory_notes=state["memory_notes"])
    except agent.MissingAPIKeyError as e:
        return (decision_badge_html("unknown", None), f"**API Key Error:** {e}",
                tool_log_html([]), trace_html([]),
                stats_html(state), gr.update(value=[]), state, gr.update())
    except Exception as e:
        return (decision_badge_html("unknown", None), f"**Agent error:** {e}",
                tool_log_html([]), trace_html([]),
                stats_html(state), gr.update(value=[]), state, gr.update())

    decision_obj = result["decision"] or {}
    if not isinstance(decision_obj, dict):
        decision_obj = {}

    # Anything unrecognised (or unparseable) falls back to a human decision.
    decision = str(decision_obj.get("decision", "escalate")).strip().lower()
    if decision not in VALID_DECISIONS:
        decision = "escalate"

    confidence      = decision_obj.get("confidence", None)
    reasoning       = decision_obj.get("reasoning", result["raw_text"])
    issue_type      = str(decision_obj.get("issue_type", "unknown"))
    standard        = decision_obj.get("standard_referenced")
    summary         = decision_obj.get("verdict_summary")
    findings        = _as_list(decision_obj.get("key_findings"))
    recommendations = _as_list(decision_obj.get("recommendations"))
    reorder_qty     = _as_number(decision_obj.get("reorder_quantity"))
    discount_pct    = _as_number(decision_obj.get("discount_percent"))

    state["seq"] += 1
    record = {
        "id":               f"G-{1000 + state['seq']}",
        "time":             datetime.now().strftime("%H:%M:%S"),
        "item":             item_name,
        "description":      full_description,
        "issue_type":       issue_type,
        "decision":         decision,
        "confidence":       confidence,
        "reorder_quantity": reorder_qty,
        "discount_percent": discount_pct,
        "reasoning":        reasoning,
        "recommendations":  recommendations,
        "standard":         standard,
        "human_review":     "",
    }
    state["log"].append(record)

    badge = decision_badge_html(decision, confidence,
                                action_detail(decision, reorder_qty, discount_pct))

    parts = []
    if summary:
        parts.append(f"**{summary}**")
    parts.append(f"**Issue type:** `{issue_type}`")
    parts.append(str(reasoning))
    if findings:
        parts.append("**Key findings**\n\n" + _bullets(findings))
    if recommendations:
        parts.append("**Recommended next steps**\n\n" + _bullets(recommendations))
    if standard:
        parts.append(f"> 📚 **Guideline referenced:** {standard}")
    reasoning_md = "\n\n".join(parts)

    review_choices = [r["id"] for r in state["log"] if r["human_review"] == ""]
    return (
        badge,
        reasoning_md,
        tool_log_html(result["tool_log"]),
        trace_html(result.get("trace", [])),
        stats_html(state),
        gr.update(value=build_log_rows(state)),
        state,
        gr.update(choices=review_choices, value=(review_choices[-1] if review_choices else None)),
    )


def submit_correction(item_id, human_decision, correction_note, state):
    state = state or new_state()
    if not item_id:
        return ("""<div style="font-family:'Inter',sans-serif;color:#64748B;
                font-size:13px;padding:12px;">Select an item to review first.</div>""",
                gr.update(), state, gr.update(choices=[]))

    record = next((r for r in state["log"] if r["id"] == item_id), None)
    if record is None:
        return "Item not found.", gr.update(), state, gr.update(choices=[])

    record["human_review"] = human_decision
    if human_decision == record["decision"]:
        note = (
            f"Item '{record['item']}' — agent decided {record['decision'].upper()}, "
            f"store staff confirmed it."
        )
    else:
        note = (
            f"Item '{record['item']}' — agent decided {record['decision'].upper()}, "
            f"store staff corrected to {human_decision.upper()}."
        )
    if correction_note.strip():
        note += f" Note: {correction_note.strip()}"
    state["memory_notes"] = (state["memory_notes"] + "\n" + note).strip()

    remaining = [r["id"] for r in state["log"] if r["human_review"] == ""]
    status = f"""<div style="
        font-family:'Inter',sans-serif; color:#34D399; font-size:13px;
        background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
        border-radius:10px; padding:12px 16px;
    ">✓ Correction recorded for <strong>{item_id}</strong>.
    Agent memory updated — future reviews will reflect this.</div>"""

    return (status, gr.update(value=build_log_rows(state)), state,
            gr.update(choices=remaining, value=(remaining[-1] if remaining else None)))


def export_csv(state):
    state = state or new_state()
    if not state["log"]:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "time", "item", "description", "issue_type", "decision",
                     "confidence", "reorder_quantity", "discount_percent", "reasoning",
                     "recommendations", "standard_referenced", "human_review"])
    for r in state["log"]:
        writer.writerow([
            r["id"], r["time"], r["item"], r["description"], r["issue_type"],
            r["decision"], r["confidence"],
            "" if r["reorder_quantity"] is None else r["reorder_quantity"],
            "" if r["discount_percent"] is None else r["discount_percent"],
            r["reasoning"], " | ".join(r["recommendations"]),
            r["standard"] or "", r["human_review"],
        ])
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    tmp.write(buf.getvalue())
    tmp.close()
    return tmp.name


def reset_all():
    s = new_state()
    return (s, decision_badge_html("unknown", None), "",
            tool_log_html([]), trace_html([]), stats_html(s),
            gr.update(value=[]), gr.update(choices=[], value=None),
            freshness_gauge_html(None, None))


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium dark theme
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #07090F !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

.gradio-container {
    max-width: 1420px !important;
    margin: 0 auto !important;
    padding: 20px 28px 40px !important;
}

footer { display: none !important; }

/* Inputs */
.gr-textbox textarea,
.gr-textbox input,
input[type="text"],
textarea {
    background: #0C1525 !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 12px 14px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.13) !important;
    outline: none !important;
    background: #0F1A2E !important;
}

label > span, .label-wrap span {
    color: #94A3B8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #0284C7 0%, #7C3AED 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 20px !important;
    cursor: pointer !important;
    letter-spacing: 0.3px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    width: 100% !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(56,189,248,0.3) !important;
}

button.primary:active { transform: translateY(0) !important; }

/* Secondary buttons */
button.secondary {
    background: rgba(12,21,37,0.9) !important;
    border: 1px solid rgba(99,179,237,0.22) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

button.secondary:hover {
    border-color: rgba(56,189,248,0.5) !important;
    color: #38BDF8 !important;
    background: rgba(56,189,248,0.07) !important;
}

/* Stop/reset button */
button.stop {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.28) !important;
    color: #FCA5A5 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

button.stop:hover {
    background: rgba(239,68,68,0.14) !important;
    border-color: rgba(239,68,68,0.55) !important;
}

/* Dropdown */
select, .gr-dropdown select {
    background: #0C1525 !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}

/* Dataframe / Table */
table {
    background: transparent !important;
    border-collapse: collapse !important;
    width: 100% !important;
}

th {
    background: rgba(12,21,37,0.95) !important;
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 12px 14px !important;
    border-bottom: 1px solid rgba(99,179,237,0.1) !important;
    border-right: none !important;
    border-left: none !important;
}

td {
    background: rgba(12,21,37,0.55) !important;
    color: #CBD5E1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(99,179,237,0.05) !important;
    border-right: none !important;
    border-left: none !important;
}

tr:hover td { background: rgba(56,189,248,0.05) !important; }

/* Markdown */
.gr-markdown p {
    color: #94A3B8 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
}

.gr-markdown strong { color: #CBD5E1 !important; }

.gr-markdown code {
    background: rgba(56,189,248,0.1) !important;
    color: #38BDF8 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

.gr-markdown blockquote {
    border-left: 3px solid rgba(167,139,250,0.5) !important;
    padding: 4px 14px !important;
    color: #A78BFA !important;
    margin: 12px 0 !important;
    background: rgba(167,139,250,0.05) !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Accordion */
details, .gr-accordion {
    background: rgba(12,21,37,0.7) !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}

summary, .gr-accordion > .label-wrap {
    color: #94A3B8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 16px !important;
    cursor: pointer !important;
}

/* File download */
.file-preview {
    background: #0C1525 !important;
    border: 1px solid rgba(99,179,237,0.18) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
"""


# The custom CSS is designed for Gradio's dark mode, so force it on regardless of
# the visitor's OS / browser colour-scheme setting.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio theme
# ─────────────────────────────────────────────────────────────────────────────

_theme = gr.themes.Base(
    primary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Grocery Management Agent",
    theme=_theme,
    css=CSS,
    js=FORCE_DARK_JS,
) as demo:
    state = gr.State(new_state())

    # Hero header
    gr.HTML(HERO_HTML)

    # API key warning
    keys = agent.keys_configured()
    if not (keys["gemini"] and keys["tavily"]):
        missing = [f"{k.upper()}_API_KEY" for k, v in keys.items() if not v]
        gr.HTML(
            f"""<div style="font-family:'Inter',sans-serif;font-size:13px;color:#FBBF24;
            background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
            border-radius:10px;padding:12px 18px;margin-bottom:4px;">
            ⚠️ <strong>Missing API key(s): {', '.join(missing)}.</strong>
            Edit your <code>.env</code> file and restart the app.
            </div>"""
        )

    # Live stats row
    stats_panel = gr.HTML(stats_html(new_state()))

    # Main layout
    with gr.Row(equal_height=False):

        # ── LEFT: Input + Decision + Reasoning + Trace ──
        with gr.Column(scale=5, min_width=380):

            gr.HTML(section_header("📝", "Item Report"))
            item_name = gr.Textbox(
                label="Item Name",
                placeholder="e.g. Full-cream milk 1L  |  Basmati rice 5kg  |  Bananas (loose)",
            )

            with gr.Row():
                mfg_date = gr.DateTime(
                    label="📦 Manufacturing Date",
                    include_time=False,
                    type="string",
                )
                expiry_date = gr.DateTime(
                    label="⚠️ Expiry / Best-Before Date",
                    include_time=False,
                    type="string",
                )

            freshness_gauge = gr.HTML(freshness_gauge_html(None, None))

            description = gr.Textbox(
                label="Staff Report",
                placeholder=(
                    "Describe the item's situation — stock on hand, storage, how fast it sells, "
                    "and anything you noticed. Date info above will be added automatically.\n\n"
                    "e.g. '6 cartons left in aisle 3, we usually sell about 10 a day.'"
                ),
                lines=5,
            )
            run_btn = gr.Button("▶  Review Item", variant="primary")

            gr.HTML(section_header("🎯", "Agent Decision"))
            decision_out = gr.HTML(decision_badge_html("unknown", None))

            gr.HTML(section_header("🧠", "Reasoning"))
            reasoning_out = gr.Markdown(value="*Awaiting review…*")

            with gr.Accordion("🔎  Web Search Activity (Tavily)", open=False):
                tool_out = gr.HTML(tool_log_html([]))

            with gr.Accordion("📡  Agent Trace", open=False):
                trace_display = gr.HTML(trace_html([]))

        # ── RIGHT: Human review + Log + Export ──
        with gr.Column(scale=5, min_width=380):

            gr.HTML(section_header("👤", "Human Review & Override"))
            gr.Markdown(
                "If you **disagree** with the agent's decision — or it escalated — "
                "pick the item and choose the correct action below. Your correction feeds "
                "into agent memory and influences every future review in this session."
            )
            review_select = gr.Dropdown(
                label="Select item to review",
                choices=[],
                interactive=True,
            )
            with gr.Row():
                keep_btn     = gr.Button("✓  KEEP",     variant="secondary")
                restock_btn  = gr.Button("↻  RESTOCK",  variant="secondary")
                discount_btn = gr.Button("%  DISCOUNT", variant="secondary")
                remove_btn   = gr.Button("✗  REMOVE",   variant="secondary")
            correction_note = gr.Textbox(
                label="Staff note (optional)",
                placeholder="Why are you overriding? e.g. 'New delivery arrives tomorrow, no need to reorder'",
                lines=2,
            )
            correction_status = gr.HTML()

            gr.HTML(section_header("📋", "Item Review Log"))
            log_table = gr.Dataframe(
                headers=["ID", "Time", "Item", "Issue Type", "Decision", "Confidence", "Human Review"],
                datatype=["str"] * 7,
                row_count=0,
                column_count=7,
                interactive=False,
                wrap=False,
            )
            with gr.Row():
                export_btn = gr.Button("⬇  Export CSV", variant="secondary")
                reset_btn  = gr.Button("↺  Reset Session", variant="stop")
            export_file = gr.File(label="Download Review Report", visible=True)

    gr.HTML(FOOTER_HTML)

    # ── Wire callbacks ──

    # Live freshness gauge updates whenever either date changes
    mfg_date.change(freshness_gauge_html, inputs=[mfg_date, expiry_date], outputs=[freshness_gauge])
    expiry_date.change(freshness_gauge_html, inputs=[mfg_date, expiry_date], outputs=[freshness_gauge])

    run_btn.click(
        run_review,
        inputs=[item_name, description, mfg_date, expiry_date, state],
        outputs=[
            decision_out, reasoning_out,
            tool_out, trace_display,
            stats_panel, log_table,
            state, review_select,
        ],
    )

    for _decision, _btn in (
        ("keep", keep_btn),
        ("restock", restock_btn),
        ("discount", discount_btn),
        ("remove", remove_btn),
    ):
        _btn.click(
            submit_correction,
            inputs=[review_select, gr.State(_decision), correction_note, state],
            outputs=[correction_status, log_table, state, review_select],
        )

    export_btn.click(export_csv, inputs=[state], outputs=[export_file])

    reset_btn.click(
        reset_all,
        inputs=[],
        outputs=[
            state, decision_out, reasoning_out,
            tool_out, trace_display,
            stats_panel, log_table, review_select,
            freshness_gauge,
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ASGI application — FastAPI wraps Gradio so custom routes like /health work
# reliably on Render and any other ASGI host.
#
# Architecture:
#   FastAPI (handles /health + any future API routes)
#     └─ gr.mount_gradio_app mounts the Gradio Blocks at "/"
#
# Served by uvicorn (start command: python app.py).
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Grocery Management Agent")


@app.get("/health")
def health_check():
    """Render polls this endpoint to confirm the service is alive."""
    keys = agent.keys_configured()
    return JSONResponse(
        content={
            "status": "ok",
            "gemini_key_set": keys["gemini"],
            "tavily_key_set": keys["tavily"],
        },
        status_code=200,
    )


# Mount the Gradio UI at root. Must come AFTER all FastAPI routes so that
# FastAPI's own routes take precedence.
gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    # Render injects PORT automatically; default to 7860 locally.
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
