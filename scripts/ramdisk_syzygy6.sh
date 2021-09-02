#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

# It's generally a bad idea to store Syzygy tablebases in a RAM disk, because RAM usage could double
# as the files are memory-mapped. However, machines such as Cloud TPU VMs have lots of RAM and
# very little disk by default, so this may be the only option.

CHESSCOACH_DATA="${XDG_DATA_HOME-$HOME/.local/share}/ChessCoach"
CHESSCOACH_SYZYGY_SIX="${CHESSCOACH_DATA}/Syzygy6"

mkdir -p "${CHESSCOACH_SYZYGY_SIX}"
sudo mount -t tmpfs -o size=155g tmpfs "${CHESSCOACH_SYZYGY_SIX}"

echo "Syzygy configuration in installed config.toml:"
grep "syzygy" /usr/local/share/ChessCoach/config.toml

popd