pushd "%~dp0"
call ..\activate_virtual_env.cmd
..\tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-engine name=Stockfish_13 cmd=..\tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 option.Hash=4096 ^
	-each proto=uci tc=60+0.6 timemargin=5000 ^
	-games 2 ^
	-pgnout "%localappdata%\ChessCoach\tournament.pgn" ^
	-recover
popd