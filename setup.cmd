@echo off

pushd "%~dp0"
call activate_virtual_env.cmd

for /f "usebackq tokens=*" %%i in (`where.exe python`) do (
  set CHESSCOACH_PYTHONHOME=%%~dpi
  setx CHESSCOACH_PYTHONHOME %%~dpi
  goto break
)
:break

pip install -r requirements.txt

popd