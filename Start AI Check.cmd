@echo off
cd /d "%~dp0"
echo Browser mein http://127.0.0.1:4173 kholo. App chalane ke liye ye window open rakho.
node --env-file-if-exists=.env server.mjs
pause
