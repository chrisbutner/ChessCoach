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

TRAIN=gcr.io/$PROJECT_ID/chesscoach-train:$FULL_TAG

docker run -it --rm -h=$HOSTNAME $TRAIN