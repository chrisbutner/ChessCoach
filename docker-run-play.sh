#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

if [[ $# -lt 2 ]] ; then
  echo ERROR: Expected two arguments, NETWORK and VERSION, e.g. 'chesscoach1 v1'
  exit 1
fi

NETWORK=$1
VERSION=$2
FULL_TAG="${NETWORK}_${VERSION}"

PLAY=gcr.io/$PROJECT_ID/chesscoach-play:$FULL_TAG

docker run -it --rm --privileged \
  --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu,readonly \
  --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so,readonly \
  $PLAY