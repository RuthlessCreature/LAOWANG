Param(
  [string]$Config = "config.ini"
)

$ErrorActionPreference = "Stop"

Write-Host "[init] Using config: $Config"

# Note: This assumes the target database already exists.
# If you need to create database + tables from scratch, use sql/schema_mysql.sql.

python init.py --config $Config

Write-Host "[init] OK"
