@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CONFIG=config.ini
if not "%~1"=="" set CONFIG=%~1

echo [models] incremental update (smart; uses stock_daily + model_runs)
python models_update.py --config %CONFIG% --workers 16 --laowang-min-score 0 update
