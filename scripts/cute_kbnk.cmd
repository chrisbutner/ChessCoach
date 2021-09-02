pushd "%~dp0"
call ..\activate_virtual_env.cmd
..\tools\win\CuteChess\cutechess-cli.exe ^
	-engine name=ChessCoach cmd=uci.cmd ^
		option.syzygy_path=C:\syzygy ^
	-engine name=Stockfish_13 cmd=..\tools\win\stockfish_13_win_x64_bmi2\stockfish_13_win_x64_bmi2.exe ^
		option.Threads=8 option.Hash=8192 option.SyzygyPath=%localappdata%\ChessCoach\Syzygy ^
	-each proto=uci tc=60+0.6 timemargin=5000 ^
	-games 1 ^
	-pgnout "%localappdata%\ChessCoach\kbnk.pgn" ^
	-openings file=kbnk.pgn ^
	-recover
popd