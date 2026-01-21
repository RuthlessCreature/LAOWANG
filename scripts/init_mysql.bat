@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CONFIG=config.ini
if not "%~1"=="" set CONFIG=%~1

echo [init] Using config: %CONFIG%

REM Note: This assumes the target database already exists.
REM If you need to create database + tables from scratch, use sql/schema_mysql.sql.

python init.py --config %CONFIG%

echo [init] OK
