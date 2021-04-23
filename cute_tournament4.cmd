pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=Stockfish_13_2000 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2000 ^
	-engine name=Stockfish_13_2750 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2750 ^
	-engine name=Stockfish_13_2500 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2500 ^
	-engine name=Stockfish_13_2250 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=1 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2250 ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-each proto=uci st=1 timemargin=1000 ^
	-games 4 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd