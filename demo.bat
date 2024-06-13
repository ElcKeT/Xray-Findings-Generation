@echo off
setLocal enableExtensions

REM Examination before starting
if not defined DemoPath (
    echo demo.bat: ERROR:DemoPath is not defined
    goto:EOF
)

if not exist %DemoPath%\\infer.py (
    echo ERROR:Missing infer.py
    goto:EOF
)
python %DemoPath%\\infer.py