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
	-engine name=ChessCoach_student_2_256 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=2 ^
		option.search_parallelism=256 ^
		option.safety_buffer_milliseconds=50 ^
	-engine name=ChessCoach_student_2_128 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=2 ^
		option.search_parallelism=128 ^
	-engine name=ChessCoach_student_2_64 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=2 ^
		option.search_parallelism=64 ^
	-engine name=ChessCoach_student_2_32 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=2 ^
		option.search_parallelism=32 ^
	-engine name=ChessCoach_student_4_128 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=4 ^
		option.search_parallelism=128 ^
		option.safety_buffer_milliseconds=50 ^
	-engine name=ChessCoach_student_4_64 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=4 ^
		option.search_parallelism=64 ^
	-engine name=ChessCoach_student_4_32 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=4 ^
		option.search_parallelism=32 ^
	-engine name=ChessCoach_student_1_64 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=student ^
		option.search_threads=1 ^
		option.search_parallelism=64 ^
	-engine name=ChessCoach_teacher_4_64 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=teacher ^
		option.search_threads=4 ^
		option.search_parallelism=64 ^
		option.safety_buffer_milliseconds=100 ^
	-engine name=ChessCoach_teacher_4_32 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=teacher ^
		option.search_threads=4 ^
		option.search_parallelism=32 ^
		option.safety_buffer_milliseconds=100 ^
	-engine name=ChessCoach_teacher_4_16 cmd=uci.cmd ^
		option.network_weights=selfplay6a_001000000 ^
		option.network_type=teacher ^
		option.search_threads=4 ^
		option.search_parallelism=16 ^
		option.safety_buffer_milliseconds=100 ^
	-each proto=uci st=1 timemargin=50 ^
	-games 4 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd