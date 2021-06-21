pushd "%~dp0"
call activate_virtual_env.cmd
start cpp\ChessCoach.sln
code py
popd