@echo off
rem Resume the remaining answer-level evaluations (free tier: run once a day until done).
cd /d "%~dp0.."
call .venv\Scripts\activate
furwise eval run --stage stage1 --judge && furwise eval run --stage stage2 --judge
furwise eval table
