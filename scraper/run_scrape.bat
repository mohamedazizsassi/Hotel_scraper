@echo off
REM run_scrape.bat — wrapper for Windows Task Scheduler
REM Activates the venv and runs both spiders, logging everything to logs\.

cd /d "C:\Users\ASUS\Desktop\PFE\revway\scraper"
if not exist logs mkdir logs

REM Build a filesystem-safe timestamp: YYYY-MM-DD_HH-MM (locale-independent via PowerShell)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm"') do set TS=%%i
set LOGFILE=logs\run_%TS%.log

echo === run_scrape.bat starting at %TS% === > "%LOGFILE%"
call .venv\Scripts\activate.bat >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe run_scrape.py >> "%LOGFILE%" 2>&1
set RC=%ERRORLEVEL%
echo === Exit code: %RC% === >> "%LOGFILE%"
exit /b %RC%
