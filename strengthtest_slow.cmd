pushd "%~dp0"
call conda activate chesscoach
cpp\x64\Release\ChessCoachStrengthTest.exe -e "cpp/StrengthTests/STS.epd" -t 200 -s 445.23 -i -242.85
cpp\x64\Release\ChessCoachStrengthTest.exe -e "cpp/StrengthTests/ERET.epd" -t 10000
cpp\x64\Release\ChessCoachStrengthTest.exe -e "cpp/StrengthTests/Arasan20.epd" -t 10000