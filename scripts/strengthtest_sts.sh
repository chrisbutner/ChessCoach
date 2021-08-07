#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

INSTALLED_DATA="/usr/local/share/ChessCoach"

ChessCoachStrengthTest -e "${INSTALLED_DATA}/StrengthTests/STS.epd" -t 200 -s 445.23 -i -242.85

popd