robocopy %1.. %2 *.toml
if ErrorLevel 8 (exit /B 1)
robocopy %1..\py %2 *.py
if ErrorLevel 8 (exit /B 1)
robocopy %1..\js %2js\ /E
if ErrorLevel 8 (exit /B 1)
robocopy %1StrengthTests %2StrengthTests\
if ErrorLevel 8 (exit /B 1)
robocopy %1Dictionaries %2Dictionaries\ /E
if ErrorLevel 8 (exit /B 1)
robocopy %1..\tools\win\CuteChess %2 /E
if ErrorLevel 8 (exit /B 1)
robocopy %1..\tools\win\bayeselo %2 /E
if ErrorLevel 8 (exit /B 1)
robocopy %1..\tools\win\stockfish_13_win_x64_bmi2 %2 /E
if ErrorLevel 8 (exit /B 1)
exit /B 0