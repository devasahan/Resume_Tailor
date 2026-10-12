#!/bin/bash
# Set up Resume Tailor on PythonAnywhere. Run it in a PythonAnywhere Bash console:
#     bash ~/Resume_Tailor/deploy/pythonanywhere_setup.sh
# Safe to run again (after a git pull, for example): an existing .env is never changed.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RESUME_TAILOR_VENV:-$HOME/.virtualenvs/resume_tailor}"
ACCOUNT="${USER:-$(whoami)}"
DOMAIN="${RESUME_TAILOR_DOMAIN:-$ACCOUNT.pythonanywhere.com}"  # EU accounts: set RESUME_TAILOR_DOMAIN=$USER.eu.pythonanywhere.com
ENV_FILE="$APP_DIR/.env"

set_value() {  # KEY VALUE: set KEY=VALUE in .env, replacing an empty or existing line
  if grep -q "^$1=" "$ENV_FILE"; then
    sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"
  fi
}

echo "== Resume Tailor setup in $APP_DIR"

if [ "${RESUME_TAILOR_SKIP_INSTALL:-}" != "1" ]; then
  PY=""
  for version in 3.13 3.12 3.11; do
    if command -v "python$version" >/dev/null 2>&1; then PY="python$version"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.11 or newer was not found. Change the system image under Account > System image, then run this again."
    exit 1
  fi
  echo "== Python: $($PY --version)"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "== Creating the virtualenv $VENV"
    "$PY" -m venv "$VENV"
  fi
  echo "== Installing the libraries (a few minutes the first time)"
  # --no-cache-dir: the download cache would count against the free account's disk space
  "$VENV/bin/pip" install --no-cache-dir --quiet --upgrade pip
  "$VENV/bin/pip" install --no-cache-dir --quiet -r "$APP_DIR/requirements.txt" pythonanywhere
fi

if [ -f "$ENV_FILE" ]; then
  echo "== .env already exists: left as it is"
else
  cp "$APP_DIR/.env.example" "$ENV_FILE"
  set_value COOKIE_SECURE 1  # the session cookie travels only over HTTPS
  set_value FORCE_HTTPS 1    # http:// visitors are sent to https://
  echo "== Created $ENV_FILE"
fi
chmod 600 "$ENV_FILE"
mkdir -p "$APP_DIR/data"

if [ "${RESUME_TAILOR_SKIP_INSTALL:-}" != "1" ]; then
  echo "== Checking the setup"
  (cd "$APP_DIR" && "$VENV/bin/python" -m app.doctor) || true
fi

cat <<EOF

== Next steps (see DEPLOY.md for details)

1. Put your settings in .env:   nano $ENV_FILE
   Fill in ADMIN_EMAIL, ADMIN_PASSWORD and GEMINI_API_KEY. Save with Ctrl+O, Enter, then Ctrl+X.

2a. Start the site with ASGI (preferred). Create an API token first (Account > API token), open a NEW
    Bash console, then run this one line:

$VENV/bin/pa website create --domain $DOMAIN --command '$VENV/bin/uvicorn --app-dir $APP_DIR --uds \${DOMAIN_SOCKET} --proxy-headers --forwarded-allow-ips=* app.asgi:app'

    After a change:   $VENV/bin/pa website reload --domain $DOMAIN
    Logs:             /var/log/$DOMAIN.error.log  and  /var/log/$DOMAIN.server.log

2b. Or use the Web tab (if ASGI is not available on your account). Web > Add a new web app > Manual
    configuration > the same Python version. Then set:
      Source code:   $APP_DIR
      Virtualenv:    $VENV
      WSGI configuration file: replace everything in it with these 4 lines:
import sys
if "$APP_DIR" not in sys.path:
    sys.path.insert(0, "$APP_DIR")
from app.wsgi import application  # noqa: E402
    Turn on "Force HTTPS", then press Reload.

3. Open https://$DOMAIN and log in with ADMIN_EMAIL and ADMIN_PASSWORD.
EOF
