pushd "%~dp0"
call activate_virtual_env.cmd
cpp\x64\Release\ChessCoachUci.exe
popd