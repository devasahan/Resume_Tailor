@echo off
setlocal
cd /d "%~dp0"
title Resume Tailor

rem ---- find Python 3.11 or newer ----
set "PY="
for %%V in (3.14 3.13 3.12 3.11) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

rem ---- first run: create a private environment and install the libraries ----
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting things up. This takes a minute or two.
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
python -c "import fastapi, fpdf, docx" >nul 2>nul
if errorlevel 1 (
    echo Installing libraries...
    python -m pip install --disable-pip-version-check -q -r requirements.txt
    if errorlevel 1 goto :fail
)
if not exist ".env" copy ".env.example" ".env" >nul

python -m app.doctor --quiet
if errorlevel 1 goto :fail

python -m app
goto :end

:nopython
echo Python 3.11 or newer is needed and was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH" in the installer.
echo Then double-click start.bat again.
goto :fail

:fail
echo.
echo Setup did not finish. Run "python -m app.doctor" for details.
pause
exit /b 1

:end
pause
