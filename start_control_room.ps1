$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Get-Command py -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue

Write-Host "Starting Control Room V9 no-map dashboard."
Write-Host "Use start_control_room.bat if PowerShell execution policy blocks this script."

if ($py) {
    & $py.Source -3 scripts\control_room_server.py
} elseif ($python) {
    & $python.Source scripts\control_room_server.py
} else {
    Write-Error "Python was not found. Install Python 3 and add it to PATH."
    exit 1
}
