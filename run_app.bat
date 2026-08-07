@echo off
title Strawhat Pirates Defect Detection System - RVCE 2026
color 0A
echo =========================================================================
echo  STRAWHAT PIRATES - REAL-TIME DEFECT DETECTION SYSTEM (RVCE 2026)
echo =========================================================================
echo.
echo [1/3] Navigating to project directory...
cd /d "%~dp0"

echo [2/3] Starting Inspection API Server (Uvicorn @ http://localhost:8000)...
echo       - Auto-detects ESP32 Microcontroller on COM port
echo       - Auto-detects Iriun / USB Webcam on Port 1
echo.

start "" "http://localhost:8000"

C:\Users\mmddf\.conda\envs\rvce\python.exe -m uvicorn inference.api_server:app --host 0.0.0.0 --port 8000

pause
