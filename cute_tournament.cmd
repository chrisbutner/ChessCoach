pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=Stockfish_13_2000 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2000 ^
	-engine name=AnMon_5.75 cmd=tools\win\AnMon\AnMon_5.75.exe ^
	-engine name=ChessCoach_200 cmd=uci.cmd ^
		option.network_weights=selfplay6a_000200000 ^
	-engine name=ChessCoach_400 cmd=uci.cmd ^
		option.network_weights=selfplay6a_000400000 ^
	-engine name=ChessCoach_600 cmd=uci.cmd ^
		option.network_weights=selfplay6a_000600000 ^
	-engine name=ChessCoach_800 cmd=uci.cmd ^
		option.network_weights=selfplay6a_000800000 ^
	-engine name=ChessCoach_1000 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
	-each proto=uci st=1 timemargin=25 ^
	-games 4 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd