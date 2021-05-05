#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"

cutechess-cli \
	-engine name=Stockfish_13 cmd=tools/deb/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2 \
		option.Threads=4 \
	-engine name=Stockfish_13_2850 cmd=tools/deb/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2 \
		option.Threads=1 \
		option.UCI_LimitStrength=true \
		option.UCI_Elo=2850 \
	-engine name=ChessCoach cmd=ChessCoachUci \
	-each proto=uci tc=40/5:00 timemargin=1000 \
	-games 4 \
	-pgnout "${CHESSCOACH_DATA}/tournament.pgn" \
	-recover

popd