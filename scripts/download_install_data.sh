#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"
CHESSCOACH_DATA_ZIP="${CHESSCOACH_DATA}/Data.zip"

mkdir -p "${CHESSCOACH_DATA}"

curl -L https://github.com/chrisbutner/ChessCoachData/releases/download/v1.0/Data.zip -o "${CHESSCOACH_DATA_ZIP}"

unzip -o "${CHESSCOACH_DATA_ZIP}" -d "${CHESSCOACH_DATA}"

rm "${CHESSCOACH_DATA_ZIP}"

popd