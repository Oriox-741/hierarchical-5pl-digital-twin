@echo off
setlocal
cd /d "%~dp0"

set "PY_RUN="
where py > nul 2> nul && set "PY_RUN=py -3"

if not defined PY_RUN (
  where python > nul 2> nul && set "PY_RUN=python"
)

if not defined PY_RUN (
  echo Python was not found.
  echo Install Python 3 and add it to PATH.
  pause
  exit /b 1
)

echo Starting Control Room V9 no-map dashboard.
echo This window stays open while the server is running.
%PY_RUN% scripts\control_room_server.py
echo.
echo Control Room server exited.
pause
exit /b 0
