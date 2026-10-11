# Resume Tailor

A resume tailoring app that works differently from the usual one.

- Your **base resume holds only fixed facts**: name, contact details, employers (with dates and remote/onsite) and education. Code locks these. The AI never writes or changes them.
- The AI **writes everything else from the job description**: headline, summary, skills, a job title per employer (the titles form a believable career path toward the target role) and the bullets under each employer.
- The result is a **one-page, single-column, ATS-friendly resume** (PDF and DOCX), plus an optional matching cover letter.
- It is **multi-model**: pick any model per run, tick several to compare them side by side, set defaults and fallbacks, and change models at any time without touching code.

> Everything the AI writes is a draft. It cannot know your real history, so the app flags every number and every inferred skill for you to confirm. Do not send a resume you have not read.

## Quick start

Requires Python 3.11+. WeasyPrint needs the Pango library (Debian/Ubuntu: `apt install libpango-1.0-0 libpangoft2-1.0-0`; macOS: `brew install pango`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add the API keys you have
python -m app                 # http://127.0.0.1:8000
```

With **no API keys** the app still runs end to end using three offline demo models. Their text is template-based and says nothing about what real models write; they exist so you can try the whole flow.

### Using it
1. **Profile**: upload your base resume (PDF, DOCX, TXT) or type the facts in. Check the fields, then save. By default the file is read by rules on your machine; you can choose a model to read it instead (then the resume text is sent to that provider).
2. **Tailor**: paste a job description. Optionally add *real facts about you* (years, tools you actually used, real numbers); the AI treats these as true and invents nothing beyond them. Pick **One model**, or **Compare several**.
3. **Review and edit**: change any AI-written text. A live PDF preview shows whether it still fits on one page. Items to confirm are listed.
4. **Export** PDF or DOCX for the resume and the cover letter.

## Models

`models.yaml` is the registry. A model is usable when its API-key variable is set (see `.env.example`) and its data region is allowed in Settings.

| Provider type | Used for | How |
|---|---|---|
| `anthropic` | Claude Sonnet 5.5 | Official Anthropic SDK; JSON schema via `output_config.format`, adaptive thinking, refusal fallback |
| `openai_compat` | Gemini, Grok, Mistral, DeepSeek, Qwen, GLM, Kimi | One adapter for any `POST {base_url}/chat/completions` API. If a provider rejects `json_schema` or `reasoning_effort`, it retries with fewer features and remembers what worked |
| `fake` | Offline demo models | No network, no cost |

**To add a model**, add an entry to `models.yaml` (id, model string, base URL, key variable, price, region), set the key, restart. No code change is needed for any OpenAI-compatible provider.

**Models marked `verified: false` need checking.** Their model strings and prices came from third-party websites in October 2026 and were not confirmed against the vendors. Run:

```bash
python -m app.cli models          # what is available
python -m app.cli check           # one tiny real call per model (cents): verifies key, model name, JSON mode
```

Costs shown in the app are estimates: `tokens reported by the provider x the price in models.yaml`. Reasoning tokens usually count as output tokens.

### Settings (applied to new runs immediately)
- default model for the resume, the cover letter and for reading the base resume
- fallback models, tried in order when the chosen one fails (not used in Compare, where each result must show one model alone)
- a **soft daily cost cap** (new runs stop once today's recorded spend reaches it; runs already in flight finish)
- allowed data regions (models hosted in `CN` or `UNKNOWN` regions stay off until you allow them)
- retention (stored profile and runs are deleted after N days), paper size (Letter / A4)

## Choosing a model: the bake-off

Nobody can tell you which model writes your resumes best; measure it.

```bash
# put 10-20 real job descriptions in evals/jds/*.txt, your own facts in evals/my_profile.json
python -m app.cli bakeoff --jds evals/jds --profile evals/my_profile.json \
    --models claude-sonnet-5-5,gemini-3.8-flash,mistral-large --out out/bakeoff
```

It prints a cost estimate and asks before spending anything, then writes `results.csv`, `summary.md` and one PDF per job and model. Compare the PDFs blind. The summary reports one-page rate, keyword match, how many unverified numbers each model invented, cost and speed. It does not score writing quality. The same comparison is available in the browser under Tailor > Compare several.

## How the guarantees work
- **Fixed facts cannot change.** The model returns only the editable sections as JSON. Code merges them into your saved profile and asserts the profile is unchanged. The edit endpoint never reads profile fields from the request.
- **No invented numbers.** The prompts forbid figures you did not supply. Anything numeric the model writes anyway is flagged "to verify". Skills that are not in the job description are flagged as inferred.
- **One page.** The renderer shrinks font and margins (floor 9.5 pt / 0.5 in), then asks the model once for shorter bullets, then drops the lowest-priority bullets of the roles with the most bullets (keeping two each), and reports what it did. If it still does not fit, it says so.
- **ATS-friendly layout.** Single column, real text, standard fonts, standard section headings, no tables, columns, images or icons, contact details in the body. No tool can guarantee how every ATS parses a file; this follows the common advice.
- **Bad model output is handled.** Invalid or incomplete JSON is sent back once with the error; if it still fails the next fallback model is used; failures show up in History instead of crashing.

## Privacy and security
- Your resume text and job description go **only to the model you pick**, over HTTPS, with your own API key. Check each provider's retention terms; use paid keys (free tiers may let the provider train on your data).
- Stored locally in SQLite (`data/`, file mode 600): your profile, generated resumes and cover letters, and the "real facts" text you typed for each run (kept so the number checks still work after you edit). Job descriptions are **not stored** (only a hash). Everything is deleted after the retention period, or immediately via Settings > Delete all stored data.
- API keys come from environment variables only and are never returned by the API or stored in the database.
- This is a **single-user app with no accounts**. Run it on your own machine. If you host it, put real authentication in front of it. Set `ADMIN_TOKEN` to protect settings changes and data deletion.
- Uploads are limited to 5 MB, 10 PDF pages and a bounded DOCX unpack size; the browser UI inserts all text with `textContent`, so model output cannot inject markup.

## Development

```bash
pip install pytest && python -m pytest        # no network or API keys needed
```

Tests cover the schemas, validation and fixed-field integrity, both provider adapters (against mocked HTTP and a fake SDK client), routing, fallback and cost cap, the full pipeline with the demo models, every API endpoint and the upload hardening. The browser UI was also exercised end to end with headless Chromium.

**What is not verified:** calls to the real provider APIs (no keys were available while building), the exact model strings and prices marked unverified, and writing quality for any real model. Use `app.cli check` and the bake-off for those.

Known limits: scanned (image-only) PDFs are not supported (no OCR); the rule-based reader is rough on unusual layouts, which is why you always review the profile; the page-fit logic is tuned for Letter/A4 resumes of roughly 3-5 employers.

```
models.yaml            model registry (prices, regions, key variables, settings)
app/llm/               adapters (anthropic, openai_compat, fake), registry, router, schema helpers
app/pipeline/          tailor (whole run), generate (validate + retry + fallback), validate, fit, compare
app/prompts/           versioned prompt files
app/render/            HTML -> PDF (WeasyPrint) and DOCX
app/parsing/           PDF/DOCX/TXT -> text -> profile
app/web/               pages (Profile, Tailor, Review, History, Settings)
app/cli.py             models, check, bakeoff
evals/                 sample profile and job descriptions
```
