# Resume Tailor

A resume tailoring app that works differently from the usual one.

- Your **base resume holds only fixed facts**: name, contact details, employers (with dates and remote/onsite) and education. Code locks these. The AI never writes or changes them.
- The AI **writes everything else from the job description**: headline, summary, skills, a job title per employer (the titles form a believable career path toward the target role) and the bullets under each employer.
- The result is a **one-page, single-column, ATS-friendly resume** (PDF and DOCX), plus an optional matching cover letter.
- It is **multi-model**: pick any model per run, tick several to compare them side by side, set defaults and fallbacks, and change models at any time without touching code.

> Everything the AI writes is a draft. It cannot know your real history, so the app flags every number and every inferred skill for you to confirm. Do not send a resume you have not read.

## Run it on your PC

You need **Python 3.11 or newer** and nothing else: no API key, no other software. The app runs in **demo mode** (three offline demo models) until you add keys, so you can try the whole flow first.

| Your computer | Do this |
|---|---|
| **Windows** | 1. Install Python from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** (or run `winget install -e --id Python.Python.3.12`).<br>2. Get the project (green **Code** button > **Download ZIP**, then unzip, or `git clone`).<br>3. Double-click **`start.bat`**. |
| **macOS / Linux** | 1. Python 3.11+ (macOS: <https://www.python.org/downloads/> or `brew install python`; Ubuntu/Debian: `sudo apt install python3 python3-venv`).<br>2. In the project folder run **`./start.sh`**. |
| **By hand** | `python -m venv .venv`, activate it (`source .venv/bin/activate` or `.venv\Scripts\activate`), `pip install -r requirements.txt`, `python -m app` |

The first start takes a minute or two (it creates a private `.venv` folder and installs the libraries). Your browser then opens at <http://127.0.0.1:8000>. Press **Ctrl+C** (or close the window) to stop it.

If something does not start, run **`python -m app.doctor`**. It checks Python, the libraries, PDF and DOCX export, the data folder, API keys and the port, and says how to fix each problem.

### Try the demo (about two minutes)
1. **Profile** > choose `evals/sample_base_resume.docx` (or `.pdf` / `.txt`) > **Read file** > check the fields > **Save profile**.
2. **Tailor** > paste a job description (for example `evals/jds/01_data_engineer.txt`) > tick **Compare several** > tick the three *Demo* models > **Tailor resume**.
3. Click **Download PDF** on a result card, or open **Review and edit** to change text with a live PDF preview first.
   Files are named company, your name, role: `Acme_Corp_Jordan_Rivera_Senior_Data_Engineer.pdf`. The company is the one you type in
   **Company applying to**, or the one the job description names; it is left out if neither has one.

The demo models write generic template text. They exist to show the flow (compare, fit to one page, flags for invented numbers, retries); they say nothing about what real models write.

### Then add real models
1. Copy `.env.example` to `.env` (the start script does this for you) and open `.env` in any text editor.
2. Paste the API key for the provider you have, for example `ANTHROPIC_API_KEY=...`. Use **paid** keys: free tiers may let the provider train on your data.
3. Stop and start the app again. In **Settings**, press **Run check** to confirm each key and model name works (one tiny call per model, a fraction of a cent).
4. In **Tailor**, pick the model, or compare several. Costs are shown per run.

Nothing leaves your computer except the calls to the model providers you pick. See [Privacy and security](#privacy-and-security).

## Using it
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
- numbers in bullets: realistic estimates flagged for you to confirm (default), or only the numbers you give

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
- **Years of experience are calculated, not guessed.** The app computes them from your locked employment dates (overlapping jobs counted once, rounded down, e.g. "7+ years") and gives that figure to the model, which must use it exactly in the summary.
- **Metrics are estimates until you confirm them.** By default (Settings > Numbers in bullets) the model adds realistic, conservative metrics to most bullets, using your real numbers from "Real facts about you" first. Every figure you did not give is flagged "to verify" in the review panel so you can replace or remove it. Switch the setting to "Only the numbers I give" to forbid estimates.
- **Related skills are marked.** The skills list starts with the job description's skills, then adds closely related skills that support them; each added one is flagged so you keep only what you really have.
- **One page.** The renderer shrinks font and margins (floor 9.5 pt / 0.5 in), then asks the model once for shorter bullets, then drops the lowest-priority bullets of the roles with the most bullets (keeping two each), and reports what it did. If it still does not fit, it says so.
- **ATS-friendly layout.** Single column, real text with an embedded standard-metrics font (Carlito, compatible with Calibri), standard section headings, no tables, columns, images or icons, contact details in the body. No tool can guarantee how every ATS parses a file; this follows the common advice.
- **Bad model output is handled.** Invalid or incomplete JSON is sent back once with the error; if it still fails the next fallback model is used; failures show up in History instead of crashing.

## Privacy and security
- Your resume text and job description go **only to the model you pick**, over HTTPS, with your own API key. Check each provider's retention terms; use paid keys (free tiers may let the provider train on your data).
- Stored locally in SQLite (`data/`, file mode 600): your profile, generated resumes and cover letters, and the "real facts" text you typed for each run (kept so the number checks still work after you edit). Job descriptions are **not stored** (only a hash). Everything is deleted after the retention period, or immediately via Settings > Delete all stored data.
- API keys come from environment variables or your local `.env` only and are never returned by the API or stored in the database. `.env` is excluded from git.
- This is a **single-user app with no accounts** and it listens on `127.0.0.1` (your computer only) by default. If you host it anywhere else, put real authentication in front of it. Set `ADMIN_TOKEN` to protect settings changes and data deletion.
- Uploads are limited to 5 MB, 10 PDF pages and a bounded DOCX unpack size; the browser UI inserts all text with `textContent`, and pages are served with a content security policy.

## Troubleshooting

| Problem | What to do |
|---|---|
| Window closes immediately / "Python not found" | Install Python 3.11+ (tick "Add python.exe to PATH" on Windows), then run `start.bat` again |
| Anything else that will not start | `python -m app.doctor` says what is wrong and how to fix it |
| "Port 8000 is already in use" | Another copy is running. Close it, or set another port: Windows `set PORT=8001`, macOS/Linux `PORT=8001 ./start.sh` |
| Browser did not open | Open <http://127.0.0.1:8000> yourself (or set `OPEN_BROWSER=0` to stop it trying) |
| PDF preview is blank | Use **Download PDF** (some browsers block inline PDFs) |
| A model says "no API key" | Put the key in `.env`, then stop and start the app again |
| A model fails on the first call | Run **Settings > Run check**: the error names the problem (key, model name, unsupported feature). Fix the entry in `models.yaml` |
| "hit the token limit" | Raise `max_output_tokens` for that model in `models.yaml` or lower its reasoning effort (reasoning tokens count too) |
| Upload says "No readable text" | The PDF is a scan (an image). Export a text PDF or DOCX from your word processor |

## Development

```bash
pip install pytest && python -m pytest        # no network or API keys needed
```

Tests cover the schemas, validation and fixed-field integrity, both provider adapters (against mocked HTTP, a local mock server, and the real Anthropic SDK pointed at a mock Messages API), routing, fallback and cost cap, the full pipeline with the demo models, the PDF/DOCX renderers, every API endpoint and the upload hardening. The browser UI was also exercised end to end with headless Chromium.

**What is not verified:** calls to the real provider APIs (no keys were available while building), the exact model strings and prices marked unverified, writing quality for any real model, and running on Windows/macOS (built and tested on Linux; the PDF engine is pure Python and `start.bat` is plain batch, but neither has been run on those systems). Use `app.doctor`, `app.cli check` and the bake-off for those.

Known limits: scanned (image-only) PDFs are not supported (no OCR); names in scripts the bundled font lacks (for example Chinese) are missing from the PDF, but not the DOCX; the rule-based reader is rough on unusual layouts, which is why you always review the profile; the page-fit logic is tuned for Letter/A4 resumes of roughly 3-5 employers.

```
models.yaml            model registry (prices, regions, key variables, settings)
start.bat / start.sh   one-click start for Windows / macOS+Linux
app/doctor.py          setup check (python -m app.doctor)
app/llm/               adapters (anthropic, openai_compat, fake), registry, router, schema helpers
app/pipeline/          tailor (whole run), generate (validate + retry + fallback), validate, fit, compare
app/prompts/           versioned prompt files
app/render/            PDF (fpdf2, pure Python, bundled Carlito font) and DOCX (python-docx)
app/parsing/           PDF/DOCX/TXT -> text -> profile
app/web/               pages (Profile, Tailor, Review, History, Settings)
app/cli.py             models, check, bakeoff
evals/                 sample base resume, sample profile and job descriptions
```

### Third-party licenses
The bundled font **Carlito** is licensed under the SIL Open Font License 1.1 (`app/render/fonts/OFL-Carlito.txt`). **fpdf2** is LGPL-3.0; the other libraries are listed in `requirements.txt` with their own permissive licenses.
