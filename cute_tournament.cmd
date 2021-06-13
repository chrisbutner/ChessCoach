pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=Stockfish_13_2850 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2850 ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-each proto=uci tc=600+6 timemargin=5000 ^
	-games 4 ^
	-pgnout "%localappdata%\ChessCoach\tournament.pgn" ^
	-recover
popd