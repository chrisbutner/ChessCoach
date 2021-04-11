set input=%~dp0bayeselo.input
set bayeselo=%~dp0tools\win\bayeselo\bayeselo.exe
pushd %localappdata%\ChessCoach
%bayeselo% < %input%
popd