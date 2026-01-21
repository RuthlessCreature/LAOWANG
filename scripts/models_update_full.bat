@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CONFIG=config.ini
if not "%~1"=="" set CONFIG=%~1

echo [models] full recompute (slow; all dates in stock_daily)
python models_update.py --config %CONFIG% --workers 16 --laowang-min-score 0 full
