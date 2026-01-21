Param(
  [string]$Config = "config.ini"
)

$ErrorActionPreference = "Stop"

Write-Host "[models] full recompute (slow; all dates in stock_daily)"
python models_update.py --config $Config --workers 16 --laowang-min-score 0 full
