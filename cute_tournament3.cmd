pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=Stockfish_13_2000 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2000 ^
	-engine name=Stockfish_13_2750 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2750 ^
	-engine name=Stockfish_13_2500 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2500 ^
	-engine name=Stockfish_13_2250 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
		option.UCI_LimitStrength=true ^
		option.UCI_Elo=2250 ^
	-engine name=ChessCoach_SBLE_PUCT_2_128 cmd=uci.cmd ^
		option.use_sble_puct=true ^
		option.search_threads=2 ^
		option.search_parallelism=128 ^
	-engine name=ChessCoach_AZ_PUCT_2_128 cmd=uci.cmd ^
		option.use_sble_puct=false ^
		option.search_threads=2 ^
		option.search_parallelism=128 ^
	-engine name=ChessCoach_AZ_PUCT_2_256 cmd=uci.cmd ^
		option.use_sble_puct=false ^
		option.search_threads=2 ^
		option.search_parallelism=256 ^
	-engine name=ChessCoach_AZ_PUCT_4_256 cmd=uci.cmd ^
		option.use_sble_puct=false ^
		option.search_threads=4 ^
		option.search_parallelism=256 ^
		option.safety_buffer_milliseconds=50 ^
	-each proto=uci st=1 timemargin=100 ^
	-games 4 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd