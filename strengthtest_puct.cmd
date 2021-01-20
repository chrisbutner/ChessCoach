pushd "%~dp0"
call conda activate chesscoach
cpp\x64\Release\ChessCoachStrengthTest.exe --epd "cpp/StrengthTests/Arasan21.epd" --network student --nodes 100000 --failure 10000000 --limit 10