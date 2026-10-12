# Put Resume Tailor online with PythonAnywhere

This guide is for a **free account on www.pythonanywhere.com**. It takes about 20 minutes. You type the commands
into a PythonAnywhere **Bash console** (Consoles > Bash), where you can paste with Ctrl+V or Shift+Insert.

In the commands below, replace `USERNAME` with your PythonAnywhere username. The setup script prints the
commands with your username already filled in.

**Before you start**
- **Free accounts** have one website, one worker, a monthly renewal and internet access to approved sites only.
  Google's Gemini API is approved. Mistral may not be, so use Gemini for cover letters too.
- **Never paste a key, token or password into a chat.** They go only in the GitHub prompt and in `.env` on
  PythonAnywhere.

## 1. A GitHub token (the code is private)
1. On GitHub: your picture > **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained
   tokens** > **Generate new token**.
2. Name: `PythonAnywhere`. Expiration: 90 days. Repository access: **Only select repositories** >
   `Resume_Tailor`. Permissions > Repository permissions > **Contents: Read-only**.
3. **Generate token** and copy it. You need it in the next step; you can make a new one any time.

## 2. Copy the code to PythonAnywhere
Open **Consoles > Bash** and run:
```
git clone -b claude/brave-pascal-6781t4 https://github.com/devasahan/Resume_Tailor.git
```
When it asks: **Username** = your GitHub username, **Password** = paste the token. The characters stay hidden
while you paste; press Enter.

## 3. Install it
```
bash ~/Resume_Tailor/deploy/pythonanywhere_setup.sh
```
It creates a private Python environment, installs the libraries (a few minutes the first time), creates
`~/Resume_Tailor/.env` with the server settings, and prints the next commands with your username in them.

## 4. Your settings in .env
```
nano ~/Resume_Tailor/.env
```
Fill in these lines. Write the value right after `=`, with no spaces and no quotes:
```
ADMIN_EMAIL=your email
ADMIN_PASSWORD=your admin password (at least 10 characters)
GEMINI_API_KEY=your Gemini key
```
`COOKIE_SECURE=1` and `FORCE_HTTPS=1` are already set. On a free account the script also adds
`HTTPS_PROXY=` and `HTTP_PROXY=` lines: the website reaches Gemini through PythonAnywhere's proxy. Save with **Ctrl+O**, **Enter**, then close with
**Ctrl+X**. (Or edit the file in the **Files** tab: `Resume_Tailor` > `.env`.)

## 5. Start the website
Choose **A** first. If PythonAnywhere says it is not available on your account, use **B**.

### A. ASGI website (preferred: several people can use it at the same time)
1. **Account** > **API token** > **Create a new API token**.
2. Open a **new** Bash console, so it knows the token.
3. If you already have a website on the **Web** tab, delete it first: a free account has one website slot.
   If that website runs another app you want to keep, do not delete it: make a second free account (another
   username and email) for Resume Tailor and do steps 2-5 there. The address is then that username's
   `.pythonanywhere.com`.
4. Run the line the setup script printed. It looks like this, all on one line:
```
~/.virtualenvs/resume_tailor/bin/pa website create --domain USERNAME.pythonanywhere.com --command '/home/USERNAME/.virtualenvs/resume_tailor/bin/uvicorn --app-dir /home/USERNAME/Resume_Tailor --uds ${DOMAIN_SOCKET} --proxy-headers --forwarded-allow-ips=* app.asgi:app'
```
If it says ASGI websites are not enabled, you can ask PythonAnywhere to switch on the beta (email
support@pythonanywhere.com), or use B now and move to A later.

### B. Web tab website (always works on a free account)
It handles one request at a time: while one person tailors a resume (about a minute), the others wait.
1. **Web** > **Add a new web app** > **Next** > **Manual configuration** > choose the Python version that the
   setup script printed (for example 3.13) > **Next**.
2. On the Web page that opens:
   - **Source code**: `/home/USERNAME/Resume_Tailor`
   - **Working directory**: `/home/USERNAME/Resume_Tailor`
   - **Virtualenv**: `/home/USERNAME/.virtualenvs/resume_tailor`
3. Click the **WSGI configuration file** link, delete everything in it, paste these 4 lines (with your
   username), and **Save**:
```
import sys
if "/home/USERNAME/Resume_Tailor" not in sys.path:
    sys.path.insert(0, "/home/USERNAME/Resume_Tailor")
from app.wsgi import application  # noqa: E402
```
4. Back on the Web tab: turn on **Force HTTPS**, then press the green **Reload** button.

## 6. Check it
1. Open **https://USERNAME.pythonanywhere.com**. You should see the **Log in** page.
2. Also open **http://USERNAME.pythonanywhere.com** (http, without the s). The address must change to
   **https://** by itself. If it stays http://, do not log in there and tell Claude.
3. Log in with `ADMIN_EMAIL` and `ADMIN_PASSWORD`.
4. **Settings** > Check models > tick **Gemini 3.8 Flash** > **Run check**: it should say "works".
5. **Settings**: Resume model = Gemini 3.8 Flash, Cover letter model = Same as the resume model > **Save settings**.
6. **Profile**: upload your base resumes. The server starts empty; your PC's resumes are not copied.
7. Test a user: open a private browser window, **Sign up** with another email, approve it on **Users**, log in
   as that user, tailor once and download. The download appears on **History**.

## Updating to a new version
In a Bash console:
```
cd ~/Resume_Tailor && git pull && ~/.virtualenvs/resume_tailor/bin/pip install --no-cache-dir -r requirements.txt
```
(`git pull` asks for your GitHub username and token again.) Then restart the website:
- **A**: `~/.virtualenvs/resume_tailor/bin/pa website reload --domain USERNAME.pythonanywhere.com`
- **B**: the **Reload** button on the Web tab.

Changing `.env` also needs this restart. A new `ADMIN_PASSWORD` replaces the old one and logs the admin out.

## When something is wrong
| Problem | What to do |
|---|---|
| The page does not load | Look at the error log: A: `/var/log/USERNAME.pythonanywhere.com.error.log` and `.server.log`; B: the **Error log** link on the Web tab |
| "Something went wrong" right after the install | Run the setup script again; it reinstalls missing libraries and keeps your `.env` |
| Log in says the admin account is not set up | `ADMIN_EMAIL` or `ADMIN_PASSWORD` is empty in `.env`; fill it in and restart the website |
| Log in keeps returning to the login page | You are on the http:// address and the site did not switch to https://. Use only https://, change `ADMIN_PASSWORD` in `.env`, restart the website, and tell Claude |
| "Cross-site request refused" | Tell Claude: the hosting proxy changes a header the app checks |
| Check models says "network error (ConnectError)" | The website does not know PythonAnywhere's proxy. In a PythonAnywhere Bash console run `echo $https_proxy`, add `HTTPS_PROXY=` and `HTTP_PROXY=` lines with that address (usually `http://proxy.server:3128`) to `.env`, and restart the website |
| Gemini check fails with "403" or "Tunnel connection failed" | The free account cannot reach that address; a paid account has full internet access |
| Gemini says "quota" or "429" | The Gemini free tier allows few requests per minute; wait a minute, or turn on billing in Google AI Studio |
| The site stopped after a month | Free accounts must be renewed: on the Web tab (B) press **Run until 1 month from today**; for A, log in to PythonAnywhere and follow its renewal notice |
