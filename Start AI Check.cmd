@echo off
cd /d "%~dp0"
where node >nul 2>&1
if errorlevel 1 (
  echo Node.js install karo aur phir ye file dobara kholo.
  pause
  exit /b 1
)
if not exist node_modules\.package-lock.json (
  call npm ci
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo Browser mein http://127.0.0.1:4173 kholo. App chalane ke liye ye window open rakho.
node server.mjs
pause
