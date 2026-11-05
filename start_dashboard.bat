@echo off
title City Brain Autonomous Dashboard Server
echo ===================================================
echo Starting City Brain Live Dashboard Server...
echo ===================================================
cd /d "%~dp0"
python src/city_brain_dashboard.py
pause
