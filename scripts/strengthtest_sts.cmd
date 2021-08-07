pushd "%~dp0"
call ..\activate_virtual_env.cmd
..\cpp\x64\Release\ChessCoachStrengthTest.exe -e "../cpp/StrengthTests/STS.epd" -t 200 -s 445.23 -i -242.85
popd