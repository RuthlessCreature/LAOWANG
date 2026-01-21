@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Windows convenience wrapper. Cross-platform entry is:
REM   python everyday.py --config config.ini

call conda activate p312
python everyday.py --config config.ini

