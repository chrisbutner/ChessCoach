pushd "%~dp0"
call activate_virtual_env.cmd

call msbuild.cmd cpp\ChessCoach.sln -t:Stockfish;hunspell;protobuf;crc32c -p:Configuration=Release -p:Platform=x64 -p:PostBuildEventUseInBuild=false -m
if %errorlevel% neq 0 exit /b
call msbuild.cmd cpp\ChessCoach.sln -t:ChessCoach -p:Configuration=Release -p:Platform=x64 -p:PostBuildEventUseInBuild=false -m
if %errorlevel% neq 0 exit /b
call msbuild.cmd cpp\ChessCoach.sln -t:ChessCoachUci;ChessCoachTest;ChessCoachTrain;ChessCoachPgnToGames;ChessCoachStrengthTest;ChessCoachGui;ChessCoachOptimizeParameters;ChessCoachBot -p:Configuration=Release -p:Platform=x64 -p:PostBuildEventUseInBuild=false -m
if %errorlevel% neq 0 exit /b

call cpp\postbuild.cmd cpp\ cpp\x64\Release\
if %errorlevel% neq 0 exit /b

robocopy cpp\x64\Release dist\ *.exe *.dll *.py *.toml
if ErrorLevel 8 (exit /B 1)

robocopy cpp\x64\Release\js dist\js\ /E
if ErrorLevel 8 (exit /B 1)

robocopy cpp\x64\Release\StrengthTests dist\StrengthTests\
if ErrorLevel 8 (exit /B 1)

robocopy cpp\x64\Release\Dictionaries dist\Dictionaries\ /E
if ErrorLevel 8 (exit /B 1)

echo ********************************************************************************
echo Installed at %~dp0dist
echo ********************************************************************************

popd