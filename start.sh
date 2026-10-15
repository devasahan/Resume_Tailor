#!/usr/bin/env bash
# One-command start for macOS and Linux:  ./start.sh
set -e
cd "$(dirname "$0")"

pick_python() {
  for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PY=$(pick_python) || { echo "Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/"; exit 1; }
  echo "First run: setting things up. This takes a minute or two."
  "$PY" -m venv .venv || { rm -rf .venv; echo "Could not create a virtual environment. On Ubuntu/Debian run: sudo apt install python3-venv"; exit 1; }
fi
. .venv/bin/activate
if ! python -c "import fastapi, fpdf, docx" 2>/dev/null; then
  echo "Installing libraries..."
  python -m pip install --disable-pip-version-check -q -r requirements.txt
fi
# Time-zone data for the Texas days (older installs lack it): worth a try, never a reason not to start
python -c "import zoneinfo; zoneinfo.ZoneInfo('America/Chicago')" 2>/dev/null \
  || python -m pip install --disable-pip-version-check -q --retries 1 --timeout 10 "tzdata>=2024.1" >/dev/null 2>&1 || true
[ -f .env ] || cp .env.example .env
python -m app.doctor --quiet
exec python -m app
