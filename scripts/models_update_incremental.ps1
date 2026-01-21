Param(
  [string]$Config = "config.ini"
)

$ErrorActionPreference = "Stop"

Write-Host "[models] incremental update (smart; uses stock_daily + model_runs)"
python models_update.py --config $Config --workers 16 --laowang-min-score 0 update
