#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"
INPUT="${PWD}/bayeselo.input"

pushd $CHESSCOACH_DATA
bayeselo < $INPUT
popd

popd