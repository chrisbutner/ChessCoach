pushd "%~dp0"
call conda activate chesscoach
tools\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish cmd=tools\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
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
	-games 2 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
	