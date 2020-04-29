robocopy %1.. %2 *.toml
if ErrorLevel 8 (exit /B 1)
robocopy %1..\py %2 *.py
if ErrorLevel 8 (exit /B 1)
robocopy %1StrengthTests %2StrengthTests\
if ErrorLevel 8 (exit /B 1)
exit /B 0