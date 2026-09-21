# Grocery Management Agent (Gemini)

An agentic AI assistant for grocery store staff. Describe an item in plain language (stock on hand, expiry date, storage, how fast it sells) and the agent decides whether to **keep**, **restock**, **discount**, **remove** or **escalate** it, with a confidence score, key findings and next steps.

When it is unsure (shelf life, food-safety guidance, recalls) it searches the web on its own before deciding.

**Stack:** Gemini (`google-genai`) for reasoning · Tavily for web search · Gradio for the UI

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# open .env and paste in your keys

python app.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

Get your keys here:

- Gemini API key: https://aistudio.google.com/apikey
- Tavily API key: https://app.tavily.com

## How it works

Each review is one full agent cycle:

1. **Perceive**: the item name and staff report, plus today's date so expiry can be judged.
2. **Reason**: Gemini weighs stock, freshness and food safety.
3. **Tool-use**: if needed, it calls `search_grocery_reference` (Tavily) for shelf-life or storage guidance, food-safety guidelines (FSSAI, FDA, USDA, Codex) or recall notices. Every search is shown in the UI.
4. **Act**: it returns a structured decision.
5. **Reflect**: if you override a decision, the correction is added to the agent's memory and applied to later reviews in the same session.

| Decision | Meaning |
| --- | --- |
| `keep` | Nothing to do; stock, freshness and storage are fine. |
| `restock` | Stock is low or will run out soon. May include a suggested reorder quantity. |
| `discount` | Safe to sell but close to expiry, overstocked or slow-moving. May include a suggested markdown %. |
| `remove` | Expired, spoiled, damaged, recalled or otherwise unsafe. |
| `escalate` | Not confident either way; a manager should decide. |

Safety comes first: anything expired or possibly unsafe can only be `remove` or `escalate`. If the model ever returns something unrecognised, the app treats it as `escalate`.

The UI also has a live stats row, an agent trace, a web-search activity panel, a review log and CSV export.

## Configuration

Set these in `.env` for local dev. On Render, add them in **Dashboard → Environment**:

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Google Gemini API key (`GOOGLE_API_KEY` is also accepted). |
| `TAVILY_API_KEY` | Recommended | Enables the web-search tool. Without it the agent still decides, but searches fail gracefully. |
| `GEMINI_MODEL` | No | Model name. Default: `gemini-3.6-flash`. |
| `GEMINI_TEMPERATURE` | No | Leave unset to use the model's default. |
| `PORT` | No | Set automatically by Render. `app.py` reads it. |

**About the model:** `gemini-3.6-flash` is the current stable Flash model (September 2026). Check Google's [model list](https://ai.google.dev/gemini-api/docs/models) if you get a 404.

**About the API:** `agent.py` uses `generateContent` through the `google-genai` SDK. The tool loop is one function (`review_item`), so switching later is a contained change.

## Project layout

```
app.py                 Gradio front-end (dark theme, date inputs, freshness gauge, stats, human review, CSV export)
agent.py               Gemini + Tavily reasoning loop (review_item)
render.yaml            Render deployment config (build/start commands, env vars, health check)
runtime.txt            Python version pin for Render
requirements.txt       Runtime dependencies
requirements-dev.txt   Adds pytest
.env.example           Template for your API keys
tests/                 Pytest suite (Gemini and Tavily are mocked)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use mocked Gemini and Tavily responses, so they need no API keys and no network.

## Deploying to Render

The repo ships a `render.yaml` that configures everything automatically.

### Option A — Blueprint (recommended, one click)

1. Push this repo to GitHub / GitLab.
2. Go to [render.com](https://render.com) → **New → Blueprint**.
3. Connect your repo. Render reads `render.yaml` and creates the service.
4. In the service dashboard → **Environment**, add:
   - `GEMINI_API_KEY` — your Google AI Studio key
   - `TAVILY_API_KEY` — your Tavily key
5. Click **Manual Deploy** (or push a commit). Done.

### Option B — Manual web service

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `python app.py` |
| Health check path | `/health` |

Then add `GEMINI_API_KEY` and `TAVILY_API_KEY` in the Environment tab.

### Notes

- `app.py` binds to `0.0.0.0` and reads `PORT` automatically — no changes needed.
- The `/health` endpoint returns `{"status":"ok"}` (HTTP 200) so Render knows the app is live.
- Never commit your `.env` file — `.gitignore` already excludes it.
- The free plan spins down after inactivity; upgrade to **Starter** for always-on.

## Customising

- **Decision rules and tone:** edit `SYSTEM_PROMPT` in `agent.py`.
- **Search behaviour:** edit the tool description in `SEARCH_TOOL`, or `run_tavily_search`.
- **Look and feel:** `CSS` and the HTML builders in `app.py`.
- **Memory:** corrections live in the browser session only and reset when you press *Reset Session* or reload. Export the CSV to keep a record.

## Troubleshooting

- **"GEMINI_API_KEY is not set"**: add the key in the Render Environment tab and redeploy.
- **"model not found" / 404**: set `GEMINI_MODEL` to a name from Google's current model list.
- **Search shows "search failed"**: check `TAVILY_API_KEY` and network connectivity.
- **Everything comes back as `escalate`**: the reply could not be parsed. Open *Agent Trace* to see what happened.
- **Render health check fails**: visit `<your-app>.onrender.com/health` — it should return `{"status":"ok"}`.


| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Google Gemini API key. `GOOGLE_API_KEY` is also accepted. |
| `TAVILY_API_KEY` | Recommended | Enables the web-search tool. Without it the agent still decides, but its searches fail gracefully. |
| `GEMINI_MODEL` | No | Model name. Default: `gemini-3.8-flash`. |
| `GEMINI_TEMPERATURE` | No | Leave unset to use the model's default. |
| `PORT` | No | Set automatically on hosts such as Render. |

**About the model:** `gemini-3.8-flash` was the newest stable Flash model in Google's model list when this project was put together (September 2026). For something cheaper, try `gemini-3.5-flash-lite`. Avoid `gemini-2.5-flash` and `gemini-2.5-flash-lite`: Google has announced they shut down on 16 October 2026. Check Google's [model list](https://ai.google.dev/gemini-api/docs/models) for current names.

**About the API:** `agent.py` uses `generateContent` through the `google-genai` SDK, which Google's docs say remains fully supported (they recommend the newer Interactions API for brand-new projects). The tool loop is one function, `review_item`, so switching later is a contained change.

## Project layout

```
app.py                 Gradio front-end (dark theme, stats, human review, CSV export)
agent.py               Gemini + Tavily reasoning loop (review_item)
requirements.txt       Runtime dependencies
requirements-dev.txt   Adds pytest
.env.example           Template for your API keys
tests/                 Pytest suite (Gemini and Tavily are mocked)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use mocked Gemini and Tavily responses (built from the real `google-genai` types), so they need no API keys and no network.

## Deploying (Render or similar)

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Environment variables: `GEMINI_API_KEY` and `TAVILY_API_KEY`

`app.py` binds to `0.0.0.0` and reads `PORT` automatically. Never commit your `.env` file; `.gitignore` already excludes it.

## Customising

- **Decision rules and tone:** edit `SYSTEM_PROMPT` in `agent.py`.
- **Search behaviour:** edit the tool description in `SEARCH_TOOL`, or `run_tavily_search`.
- **Look and feel:** `CSS` and the HTML builders in `app.py`.
- **Memory:** corrections live in the browser session only and reset when you press *Reset Session* or reload. Export the CSV if you want to keep a record.

## Troubleshooting

- **"GEMINI_API_KEY is not set"**: make sure `.env` sits next to `app.py`, then restart the app.
- **"model not found" / 404**: set `GEMINI_MODEL` to a name from Google's current model list.
- **Search shows "search failed"**: check `TAVILY_API_KEY` and your network connection.
- **Everything comes back as `escalate`**: the reply could not be parsed. Open *Agent Trace* to see what happened.
