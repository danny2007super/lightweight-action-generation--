@echo off
setlocal
set "ACT_POLICY=act"
set "ACT_SCRIPT=%~dp0start_act_live.py"
python -X utf8 "%ACT_SCRIPT%"
echo Exit code: %ERRORLEVEL%
pause
