pushd "%~dp0"
call conda activate chesscoach
tools\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish cmd=tools\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-each proto=uci st=1 timemargin=25 ^
	-games 4 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn