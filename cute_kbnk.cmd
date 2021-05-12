pushd "%~dp0"
call conda activate chesscoach
tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=ChessCoach cmd=uci.cmd ^
		option.syzygy_path=C:\syzygy ^
	-engine name=Stockfish_13 cmd=tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=4 ^
	-each proto=uci st=1 timemargin=1000 ^
	-games 1 ^
	-pgnout "%localappdata%\ChessCoach\kbnk.pgn" ^
	-openings file=kbnk.pgn ^
	-recover
popd