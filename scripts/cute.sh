#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"

cutechess-cli \
	-engine name=ChessCoach cmd=ChessCoachUci \
	-engine name=Stockfish_13 cmd=../tools/deb/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2 \
		option.Threads=8 option.Hash=8192 \
	-each proto=uci tc=60+0.6 timemargin=5000 \
	-games 2 \
	-pgnout "${CHESSCOACH_DATA}/tournament.pgn" \
	-recover

popd