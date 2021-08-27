#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

cloc ../py ../cpp/ChessCoach*

popd