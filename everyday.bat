@echo off
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set today=%datetime:~0,8%

call conda activate p312

python astock_analyzer.py run --start-date %today% --end-date %today% --workers 16
python laowang.py --output output/pool_%today%.csv --top 200 --min-score 60
python fhkq.py

pause
