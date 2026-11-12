# Resume Tailor

A resume tailoring app that works differently from the usual one.

- Your **base resume holds only fixed facts**: name, contact details, employers (with dates and remote/onsite) and education. Code locks these. The AI never writes or changes them.
- The AI **writes everything else from the job description**: headline, summary, skills, a job title per employer (the titles form a believable career path toward the target role), and the bullets under each employer, built around that company's main product during your dates there (for example a marketplace, a learning platform or an AI assistant) when the model knows the company.
- The result is a **single-column, ATS-friendly resume of up to two pages** (one for a short career; PDF and DOCX), plus an optional matching cover letter in plain words: one page under the same header as the resume (name, headline, contact line), starting right at the greeting ("Dear Hiring Manager,") and ending with a typed signature.
- **Before you tailor, see the basics of the job**: right after you paste a job post, the AI shows under the box whether it is remote, hybrid or on-site (with details such as the office days) and the salary it states, or "Not stated". It takes a few seconds and a tiny cost, and the same post pasted again costs nothing.
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

**Log in first.** The app has accounts. Open `.env` (`notepad .env` on Windows), add your own `ADMIN_EMAIL=` and `ADMIN_PASSWORD=` lines, save, and start the app again; then log in with them. That is the admin account. See [Accounts](#accounts).

If something does not start, run **`python -m app.doctor`**. It checks Python, the libraries, PDF and DOCX export, the data folder, API keys and the port, and says how to fix each problem.

### Try the demo (about two minutes)
1. **Profile** > drop `evals/sample_base_resume.docx` (or `.pdf` / `.txt`) on the page, or **Choose files**. It is read and saved at once and appears in the list.
2. **Tailor** > pick the base resume > paste a job description (for example `evals/jds/01_data_engineer.txt`) > **Create resume & cover letter**.
3. **Download resume PDF** and **Download cover letter PDF**. (**Resume only** skips the letter; **Add cover letter** writes it later for the same resume, and **Rewrite cover letter** writes a new one.)
   Files are named the usual way for applications, name then document: `Jordan_Rivera_Resume.pdf` and `Jordan_Rivera_Cover_Letter.pdf` (a name in capitals gets normal capitals).

The demo models write generic template text. They exist to show the flow (fit to two pages at most, flags for invented numbers, retries); they say nothing about what real models write.

### Then add real models
1. Copy `.env.example` to `.env` (the start script does this for you) and open `.env` in any text editor.
2. Paste the API key for the provider you have, for example `ANTHROPIC_API_KEY=...`. Or log in as the admin and paste it on **Settings > API keys**: it is saved in `.env` for you. Use **paid** keys: free tiers may let the provider train on your data.
3. A new or changed key works at once, with no restart. In **Settings**, press **Run check** to confirm each key and model name works (one tiny call per model, a fraction of a cent).
4. In **Settings**, choose the **Resume model** and **Cover letter model** the Tailor page uses (Automatic picks the first available, Gemini 3.8 Flash first when `GEMINI_API_KEY` is set). Costs are shown per run. To compare models side by side, use the bake-off below.

Nothing leaves your computer except the calls to the model providers you pick. See [Privacy and security](#privacy-and-security).

## Accounts
- **Admin**: the one account whose email and password are in `.env` (`ADMIN_EMAIL`, `ADMIN_PASSWORD`). It has every page: Profile, Tailor, **History**, **Users** and **Settings**. The password is never stored in the code or the database (only a one-way hash). To change it, edit `ADMIN_PASSWORD` and restart: the new one replaces the old one and ends every admin session.
- **Users** sign up on the **Sign up** page. A new account waits until the admin approves it on **Users** (Approve or Reject). A user sees only **Profile** (upload resumes) and **Tailor** (tailor, make a cover letter, download both PDFs): no History, Users or Settings, no model names or costs. Each account sees only its own resumes and results.
- **Users page**: approve or reject sign-ups, disable or enable an account, give a user a one-time temporary password (they change it on their **Account** page), or delete an account with its resumes and runs.
- **History** lists every click on a **Download** button, by every account (users and admin), with the account, the resume, company, job title, model and cost. Filter it by account (active accounts only; rows of disabled or removed accounts are tagged) and by typing part of a company or job title, in any case. Each row opens the job's resume, its cover letter (when it has one) and the job description in a new tab; these previews are not counted. History is deleted after the retention period.
- After 5 wrong passwords an account is locked for 15 minutes (restarting the app clears the admin's lock). The daily cost cap is shared by everyone: one busy user can reach it for the whole site.

## Using it
1. **Profile**: upload one or more base resumes (PDF, DOCX, TXT). Each is read and saved right away and listed with its employers, education, certifications and anything to double-check (for example "Phone not found"). The reader is set in Settings > Base-resume reader: Automatic uses the cheapest available model (the resume text is sent to that provider), **Rules only** reads it on your machine for free. If a read is wrong, delete it and upload again. Uploaded resumes stay until you delete them. Each one also has **Certifications you have passed**: paste the certifications this person passed, one per line as `name | issuer | date earned - date it expires` (for example `AWS Certified Developer – Associate | Amazon Web Services | Mar 2024 - Mar 2027`; rows copied from a table work too, with Issued and Expires columns; the number and exam-code columns are left out), and press **Save certifications**. LinkedIn's "Issued ... · Expires ..." lines in an uploaded resume are read the same way. While **Settings > Certifications section** is on, the best ones that fit each job, up to 4, are then printed automatically, with nothing to tick. It is off by default: then nothing is printed, and the list stays saved here for later. List only exams that were really passed: employers check them.
2. **Tailor**: pick a base resume, paste the job description, choose **US style** (the default: American spelling) or **EU style** (British spelling as used in the UK and the EU: optimisation, behaviour, centre; the letter signs off "Kind regards," and calls the resume a CV, and the download is "Download CV PDF", named "Your_Name_CV.pdf"). Names keep their own spelling (Unity Catalog, your employers), and so do the job post's own keywords, because an ATS matches them letter for letter; if the post is written in the other spelling, the Format notes say so. Then press **Create resume & cover letter**. One step, "Writing your resume and cover letter", shows the time so far: the resume and its letter are made in one go, and the letter is written from the final resume text while the app lays out the pages, so it adds little to the wait (usually about 30 seconds in all with Gemini 3.8 Flash). The resume model comes from Settings. If the letter could not be written (the AI service was busy, or the daily cost cap was reached), the resume is still ready and the page says so: press **Add cover letter** to try again.
3. The cover letter is one more call, added to the same run's cost. **Resume only** skips it; **Add cover letter** writes it later for that resume, and **Rewrite cover letter** writes a new one without tailoring the resume again. For the admin, a line under the result shows where the time went, for example "AI 21.4 s (8,214 in / 1,912 out + 2,105 thinking) · cover letter 6.1 s, alongside · layout and checks 1.3 s · total 27.7 s (this page 28.3 s)": the AI's time for the resume and its tokens (thinking counted apart), the letter's own time ("alongside" when it was written during the layout, so the wait was shorter), the app's own work, and the whole job (and what the page waited, network included). It is not shown after Add or Rewrite cover letter. Every run keeps this record, with each AI call's model, time, tokens and result.
4. **Certifications for this job** (only while **Settings > Certifications section** is on; with it off, the default, the page shows no certifications and the resume has no Certifications section): under the downloads, the AI lists the 4 best, highest-level certifications for this job, each with an **I have it** tick box. Only ticked ones are printed on the resume, in a CERTIFICATIONS section after EDUCATION, with their dates on the right ("Mar 2023 - Mar 2026": earned - expires; only the date earned when it does not expire), and the preview and downloads change at once. A certification whose expiry date has passed is never picked or printed; the Profile page lists it as expired, so you can change the date when it is renewed. The ones your uploaded resume lists (or you pasted on the Profile page) start ticked, and every tick (or untick) is remembered on that base resume for your next jobs. When you have 4 or more that fit the job, all 4 are your own. The others are ideas to earn; they are never printed unless you tick them.
5. **Download resume PDF** / **Download cover letter PDF**. There is no manual editing step: the app checks and fixes the draft itself (two pages at most, title path, required skills, thin text, figures) before you see it. History keeps the PDFs of every run.

## Models

`models.yaml` is the registry. A model is usable when its API-key variable is set (see `.env.example`, or paste the key on Settings > API keys) and its data region is allowed in Settings.

| Provider type | Used for | How |
|---|---|---|
| `anthropic` | Claude Sonnet 5.5 | Official Anthropic SDK; JSON schema via `output_config.format`, adaptive thinking, refusal fallback |
| `openai_compat` | Gemini, GPT (OpenAI), Grok, Mistral, DeepSeek, Qwen, GLM, Kimi | One adapter for any `POST {base_url}/chat/completions` API. If a provider rejects `json_schema` or `reasoning_effort`, it retries with fewer features and remembers what worked |
| `fake` | Offline demo models | No network, no cost |

**To add a model**, add an entry to `models.yaml` (id, model string, base URL, key variable, price, region), set the key, restart. No code change is needed for any OpenAI-compatible provider.

**Gemini 3.8 Flash (fast)** is the same model with the least thinking ("minimal"), for faster drafts, so the polishing pass fits the time limit more often. It is marked unverified: tick it under Check models in Settings and press Run check. If Gemini refuses the setting, the check fails (the app never quietly falls back to slower thinking); if it passes, choose it as the Resume model and compare the "job finished" lines in the server log (each AI call's line also shows its thinking tokens).

**GPT** needs an OpenAI API key (`OPENAI_API_KEY`) from platform.openai.com, paid per use: a ChatGPT Plus or Pro subscription cannot be used by apps. Its model name and price are marked unverified until the check on the Settings page passes.

**Models marked `verified: false` need checking.** Their model strings and prices came from third-party websites in October 2026 and were not confirmed against the vendors. Run:

```bash
python -m app.cli models          # what is available
python -m app.cli check           # one tiny real call per model (cents): verifies key, model name, JSON mode
```

Costs shown in the app are estimates: `tokens reported by the provider x the price in models.yaml`. Reasoning tokens usually count as output tokens.

### Settings (applied to new runs immediately)
- default model for the resume, the cover letter and for reading the base resume
- fallback models, tried in order when the chosen one fails (not used in Compare, where each result must show one model alone). A slow answer is waited for, never cut short to save time: the chosen model gets 2 minutes before the next one is tried, and the last model about 4 minutes (the connection's own limit), so a slow AI day means a longer wait, not a failed resume. A cover letter written with the resume is waited for up to 2 minutes; after that the resume comes without it, with the note to press Add cover letter. A model that was slow, busy or unreachable is asked first again in the next run, so most resumes come from the chosen model. When a key's **quota is used up** (Google: "You exceeded your current quota, please check your plan and billing details"), the app says so at once, does not retry, and rests that model for an hour, or less when the provider says when the quota resets (reload the web app to try it again sooner): check the plan and billing at aistudio.google.com (free-tier keys have low daily limits; paid keys also keep your data out of training), and pick a backup that uses **another provider's key** (for example GPT-6.1 Sol), because backups that share the key stop too: Settings warns when they do. Every failed attempt is in the server log with its reason.
- a **soft daily cost cap** (new runs stop once today's recorded spend reaches it; runs already in flight finish; the day starts at midnight in Arlington, Texas)
- allowed data regions (models hosted in `CN` or `UNKNOWN` regions stay off until you allow them)
- retention (tailored runs are deleted after N days; uploaded resumes stay until you delete them), paper size (Letter / A4)
- numbers in bullets: realistic estimates flagged for you to confirm (default), or only the numbers you give
- **Certifications section** (off by default): when on, for each job the AI picks the best 4 of your certifications (the ones saved on the Profile page) and prints them after Education, and the Tailor page lists its picks with **I have it** ticks. When off, no Certifications section is printed, the Tailor page shows no certifications, and the AI is not shown your certifications: it only lists the ones the job post names, which are never printed (a slightly shorter prompt); certification names still stay out of the skills and the text. The lists saved on the Profile page are never changed by the switch: they stay for later. Resumes made before a change keep their own section (or lack of one).
- **time limit for a resume and its cover letter** (default 30 seconds): the first draft always comes back; the extra rewrite pass (and the shorter-draft call when a draft runs over two pages) runs only when it can finish within the limit (a rewrite is expected to take at least half as long as the draft, or as long as the last one took), and is stopped at the limit, keeping the first draft with the app's own fixes (two pages, titles, dashes, spelling). With a cover letter, the time the last letter took with that model (8 seconds before the first one) is kept free as well, so the letter is ready with the resume; the trade-off is that the rewrite pass runs less often (with Gemini 3.8 Flash's 19-27 second drafts it rarely fits either way). A letter written with the resume is waited for up to 2 minutes in all (a backup letter model gets its turn after 1 minute); Add cover letter and Rewrite cover letter wait for the letter's model as long as a resume does (see fallback models). A rewrite that removes a certification named in the text always runs. 0 = no limit: always polish, best quality but slower. Each job's time per step is in the server log ("job finished in ...").

## Choosing a model: the bake-off

Nobody can tell you which model writes your resumes best; measure it.

```bash
# put 10-20 real job descriptions in evals/jds/*.txt, your own facts in evals/my_profile.json
python -m app.cli bakeoff --jds evals/jds --profile evals/my_profile.json \
    --models claude-sonnet-5-5,gemini-3.8-flash,mistral-large --out out/bakeoff
```

It prints a cost estimate and asks before spending anything, then writes `results.csv`, `summary.md` and one PDF per job and model. Compare the PDFs blind. The summary reports the within-two-pages rate, keyword match, how many unverified numbers each model invented, cost and speed. It does not score writing quality. The `/api/tailor` endpoint also takes `model` and `compare` for scripted comparisons.

## How the guarantees work
- **Fixed facts cannot change.** The model returns only the editable sections as JSON. Code merges them into your saved profile and asserts the profile is unchanged. The edit endpoint never reads profile fields from the request.
- **Years of experience are calculated, not guessed.** The app computes them from your locked employment dates (overlapping jobs counted once, rounded down, e.g. "7+ years") and gives that figure to the model, which must use it exactly in the summary.
- **Metrics are estimates until you confirm them.** By default (Settings > Numbers in bullets) the model adds realistic, specific-looking metrics to about 2 in 5 bullets (at most 2 per employer), using real numbers passed as `hints` (API and CLI only) first. A percentage never stands alone: about half of the results give only the real numbers ("cut pipeline runtime from 47 to 36 minutes", "saving $41,000 a year in cloud spend"), and every other percentage comes with its base in words ("by 41.2%, from 97 to 57", "from 847 to 669ms, a 21% drop", "by 17.8%, or $41,000 a year"). No two figures are written the same way, and a base in brackets ("by 22% (from 41 to 32 seconds)") appears at most once in the resume and once in the letter: the app rewrites any later one itself, at no cost, as the real numbers alone or in words, taking turns (a percentage you gave is always kept, and a bracket that would read awkwardly either way is left as it is), and writes 62% rather than 62.0%. Text you edit yourself is never changed. The app checks that a before-and-after base agrees with its percentage (units included: 1.2s to 818ms is 31.8%). Money is written in the currency of the country where that job was. Switch the setting to "No figures" to forbid estimates.
- **Each job opens with its biggest win.** A recruiter scans a resume for about six seconds and reads only the first bullet of each job, so that bullet is the job's strongest achievement for the posting, with its size and its result ("for 83K concurrent players, cutting queue-to-lobby time from 41 to 32 seconds"; in words under "No figures"), never a general duty. If the most recent job, or two or more jobs, open without a figure, the same single rewrite pass moves one up, and the Format notes point out anything left.
- **A certification is printed only when you have it.** The model picks the certifications that fit the job, first your own (by their number in the list it is given) and then the most respected ones you do not have yet; code prints only your own, exactly as you wrote them, and checks that every printed one is in your saved list. A certification is never a "required skill", never in the Skills list and never named in the summary or bullets (the model is told so, certification skills are removed, and a name left in the text is sent back in the one rewrite pass), so unticking one really takes it off the resume. While **Settings > Certifications section** is off (the default), nothing is printed at all: the model only lists the certifications the job post names, and those checks still keep certification names out of the skills and the text. In both modes they also catch the well-known credential abbreviations a job post names (CISSP, CCNA and the like), whatever the model lists; an abbreviation that also means something else (PMP, CISM, CISA, CSM) counts only where the post names it as a credential ("PMP required", "PMP or PRINCE2"), so "PMP deals" in an ad-tech post stays.
- **Related skills are marked.** The skills list starts with the job description's skills, then adds closely related skills that support them (frameworks, libraries and services, not only platforms); each added one is flagged so you keep only what you really have.
- **Up to two pages, richest where it matters.** Before the model writes anything, the app works out how many full-length bullets each employer gets: up to 8 / 6 / 5 / 4 / 3 by recency, fewer for short stays (2 under 6 months, 3 under a year, 5 under two years), and only as many as fit on two pages. A short career is not padded out: it stays on one page. If a draft still runs over, the app picks the largest font that fits (floor 9.5 pt / 0.5 in), and prefers one page at a slightly smaller font to a second page holding only a few lines; then it asks the model once more with one bullet fewer for the longest roles, then drops the last bullets of the roles with the most bullets (keeping two each), and reports what it did. Page breaks fall between bullets, never inside one, and a job title or section heading is never left alone at the bottom of a page (PDF and Word).
- **Strong bullets and summary.** Every bullet must say what was done (specifically), how (naming the frameworks, libraries and services the work used, not only the posting's tools), and what changed, in 20-32 words, in plain words a person would use (no "leveraged", "spearheaded", "seamless" or chains of "-ing" clauses, and not every bullet built the same way); the summary is 3-4 flowing sentences of 60-85 words. If a draft comes back thin (short summary, short bullets, "Responsible for"-style openers) or sounds machine-written (two or more such words), the model is asked once to rewrite it, and the Format notes point out anything still thin.
- **No signs of AI writing.** Following a recruiter's advice, the resume and cover letter avoid what reads as machine-written: no dashes used as punctuation (the model is told to use full stops and commas, and any dash it still writes becomes a comma; hyphenated words, ranges such as 2019-2021 and command-line flags stay), no vague results such as "improved operational efficiency" or "increased decision making", no percentage without its base, and no spelling that does not fit the market (see US style / EU style under **Using it**). Two or more vague results or bare percentages in a draft are rewritten in the one rewrite pass; a single one, and anything in the cover letter, is shown in the Format notes. Text changed through the editing API is kept as typed, with the same notes.
- **Every required skill is covered, in the posting's own words.** An ATS matches words, not meaning, so the model must show each must-have skill from the job description word for word ("schema design", not "schema changes") in the summary or a bullet, at the most plausible employer, without you supplying facts. A skill that is only listed under Skills does not count. If the first draft misses any, the same single rewrite pass adds them (the model sees its draft and changes only what is needed).
- **Keywords match the posting; sentences never copy it.** The required skills are keywords as people list them on a resume ("SQL", "AWS", "React"), never pieces of the posting's sentences ("advanced SQL statements", "AWS-hosted systems"); a list of examples ("AI tools like Copilot, Cursor, or Claude Code") becomes the tools themselves. Around each keyword, the summary and bullets say what you did in their own words. The app finds anything copied word for word from the job description (four or more words in a row, in any tense) and skills written as phrases; two or more copies, or any phrase skill, are fixed in the same single rewrite pass, and the Format notes point out anything left.
- **A career that builds toward the job.** Before writing, the model plans what each job will show from the job description: older jobs show the foundations of what the posting asks for, each later job adds scope (implemented, built, designed, owned, led), and the most recent job covers the posting's core responsibilities. Each job's bullets describe work on that company's main product or service during those dates, using its real public product names when the model knows the company; for a company it does not know, the business stays generic and no product names are invented.
- **Only tools that were really in common use.** A job mentions a tool only if it was common at work during that job, not merely released (for example GitHub Actions from 2020, OpenTofu from 2024). AI tools (LLMs, ChatGPT, AI coding assistants, agentic development) appear only at the one or two most recent jobs, and never before they became popular (LLMs and AI coding assistants 2023, agentic development 2025). The app checks this itself and has the model move such a tool in the one rewrite pass.
- **A career path that shows growth and never steps down.** A title is never lower than the one before it (no "Software Engineer II" followed by plain "Software Engineer"): the app corrects that itself at no cost, and any other title problem is fixed in the one rewrite pass.
- **A career path that shows growth.** The model is told how long each stay was and how much experience came before it, so a stay of 2+ years shows a step up (for example "Software Engineer II") and the same title never repeats three times in a row; the Format notes warn when it does. Long stays also get more bullets than short ones (a 3-year role before an 8-month one).
- **ATS-friendly layout, built for a quick scan.** The headline under your name is large and names only the role's short, common name ("Full-Stack Engineer"), not the posting's full title or a level: your domain and tools are in the summary, Skills and the bullets. A special posting title is never copied: for a "Voice AI Engineer" post the headline, the summary and your job titles say "AI Engineer", and the voice work shows in the bullets. Each job starts with the company in bold and the dates on the right, with the job title in italics on the line below; when two pages are completely full, the title moves onto the company's line instead, so the layout never costs a bullet, a page or a font size. Single column, real text with an embedded standard-metrics font (Carlito, compatible with Calibri), standard section headings, no tables, columns, images or icons, contact details in the body. No tool can guarantee how every ATS parses a file; this follows the common advice.
- **Bad model output is handled.** Invalid or incomplete JSON is sent back once with the error; if it still fails the next fallback model is used; failures show up in History instead of crashing.

## Privacy and security
- Your resume text and job description go **only to the model you pick**, over HTTPS, with your own API key. Check each provider's retention terms; use paid keys (free tiers may let the provider train on your data).
- Stored locally in SQLite (`data/`, file mode 600): your uploaded base resumes (the facts read from them, not the files), and each run's generated resume, cover letter and the job description it was made for (kept so a cover letter can be written later). Runs are deleted after the retention period; uploaded resumes stay until you delete them; Settings > Delete all stored data removes everything at once.
- API keys come from environment variables or your local `.env` only and are never returned by the API or stored in the database. `.env` is excluded from git. The admin can paste a key on **Settings > API keys**: it is written to `.env` (a new `.env` is readable by its owner only), and the page shows only the provider's marker and the key's last 4 characters, such as `sk-proj-...1234`. A key changed in `.env` by hand is also used at once.
- Every page needs a login. Passwords are stored as scrypt hashes; the session cookie is HttpOnly and SameSite, and is sent only over HTTPS when the site uses HTTPS (set `COOKIE_SECURE=1` on a server). Requests from other websites that change data are refused. By default the app listens on `127.0.0.1` (your computer only).
- **Hosting it online**: see **[DEPLOY.md](DEPLOY.md)** for PythonAnywhere, step by step. On any server, put a `.env` there with `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `COOKIE_SECURE=1`, `FORCE_HTTPS=1` and your API keys, and run `uvicorn app.asgi:app` (or `app.wsgi:application` on a WSGI host). PythonAnywhere free accounts can only reach allow-listed websites (Gemini works; not every provider does).
- Uploads are limited to 5 MB, 10 PDF pages and a bounded DOCX unpack size; the browser UI inserts all text with `textContent`, and pages are served with a content security policy.

## Troubleshooting

| Problem | What to do |
|---|---|
| Window closes immediately / "Python not found" | Install Python 3.11+ (tick "Add python.exe to PATH" on Windows), then run `start.bat` again |
| Anything else that will not start | `python -m app.doctor` says what is wrong and how to fix it |
| "Port 8000 is already in use" | Another copy is running. Close it, or set another port: Windows `set PORT=8001`, macOS/Linux `PORT=8001 ./start.sh` |
| Browser did not open | Open <http://127.0.0.1:8000> yourself (or set `OPEN_BROWSER=0` to stop it trying) |
| PDF preview is blank | Use **Download resume PDF** (some browsers block inline PDFs) |
| A model says "no API key" | Paste the key on **Settings > API keys** and press **Save** (or put it in `.env`). It works at once |
| A model fails on the first call | Run **Settings > Run check**: the error names the problem (key, model name, unsupported feature). Fix the entry in `models.yaml` |
| "hit the token limit" | Raise `max_output_tokens` for that model in `models.yaml` or lower its reasoning effort (reasoning tokens count too) |
| Upload says "No readable text" | The PDF is a scan (an image). Export a text PDF or DOCX from your word processor |

## Development

```bash
pip install pytest && python -m pytest        # no network or API keys needed
```

Tests cover the schemas, validation and fixed-field integrity, both provider adapters (against mocked HTTP, a local mock server, and the real Anthropic SDK pointed at a mock Messages API), routing, fallback and cost cap, the full pipeline with the demo models, the PDF/DOCX renderers, every API endpoint and the upload hardening. The browser UI was also exercised end to end with headless Chromium.

**What is not verified:** calls to the real provider APIs (no keys were available while building), the exact model strings and prices marked unverified, writing quality for any real model, and running on Windows/macOS (built and tested on Linux; the PDF engine is pure Python and `start.bat` is plain batch, but neither has been run on those systems). Use `app.doctor`, `app.cli check` and the bake-off for those.

Known limits: scanned (image-only) PDFs are not supported (no OCR); names in scripts the bundled font lacks (for example Chinese) are missing from the PDF, but not the DOCX; the rule-based reader is rough on unusual layouts (check the warnings on the Profile list, or choose an AI reader in Settings); the page-fit logic is tuned for Letter/A4 resumes of up to two pages; with about 7 or more employers the two pages, not the per-job limits, decide how many bullets older jobs get. With the Certifications section on (Settings), the AI picks certifications for each job, but only the ones you have (from your uploaded resume, or ticked on the Tailor page) are printed: it has no way to know which ones you hold.

```
models.yaml            model registry (prices, regions, key variables, settings)
start.bat / start.sh   one-click start for Windows / macOS+Linux
app/doctor.py          setup check (python -m app.doctor)
app/llm/               adapters (anthropic, openai_compat, fake), registry, router, schema helpers
app/pipeline/          tailor (whole run), generate (validate + retry + fallback), validate, fit, compare
app/prompts/           versioned prompt files
app/render/            PDF (fpdf2, pure Python, bundled Carlito font) and DOCX (python-docx)
app/parsing/           PDF/DOCX/TXT -> text -> profile
app/web/               pages (Profile, Tailor, History, Settings)
app/cli.py             models, check, bakeoff
evals/                 sample base resume, sample profile and job descriptions
```

### Third-party licenses
The bundled font **Carlito** is licensed under the SIL Open Font License 1.1 (`app/render/fonts/OFL-Carlito.txt`). **fpdf2** is LGPL-3.0; the other libraries are listed in `requirements.txt` with their own permissive licenses.
