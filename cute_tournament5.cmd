pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13_2500 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2500 ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-each proto=uci st=1 timemargin=1000 ^
	-games 10 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd