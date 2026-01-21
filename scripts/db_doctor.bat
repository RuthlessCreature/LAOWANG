@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CONFIG=config.ini
if not "%~1"=="" set CONFIG=%~1

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0db_doctor.ps1" -Config "%CONFIG%"

