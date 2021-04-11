pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-engine name=ChessCoach cmd=uci.cmd ^
	-each proto=uci st=1 timemargin=25 ^
	-games 2 ^
	-pgnout %localappdata%\ChessCoach\tournament.pgn
popd