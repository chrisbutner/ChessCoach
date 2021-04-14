#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"

tools/deb/cutechess-cli/cutechess-cli \
	-engine name=Stockfish_13 cmd=tools/deb/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2 \
		option.Threads=4 \
	-engine name=ChessCoach cmd=ChessCoachUci \
	-each proto=uci st=1 timemargin=25 \
	-games 2 \
	-pgnout ${CHESSCOACH_DATA}/tournament.pgn

popd