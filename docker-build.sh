#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

if [[ $# -lt 2 ]] ; then
  echo ERROR: Expected two arguments, NETWORK and VERSION, e.g. 'chesscoach1 v1'
  exit 1
fi

NETWORK=$1
VERSION=$2
BASE_TAG=$VERSION
FULL_TAG="${NETWORK}_${VERSION}"

BASE=gcr.io/$PROJECT_ID/chesscoach-base:$BASE_TAG
TRAIN=gcr.io/$PROJECT_ID/chesscoach-train:$FULL_TAG
PLAY=gcr.io/$PROJECT_ID/chesscoach-play:$FULL_TAG

BUILD_ARGS="--build-arg PROJECT_ID=${PROJECT_ID} --build-arg BASE_TAG=${BASE_TAG} --build-arg NETWORK=${NETWORK} --build-arg BUILDKIT_INLINE_CACHE=1"

DOCKER_BUILDKIT=1

docker build -t $BASE --cache-from $BASE -f base.dockerfile $BUILD_ARGS .
docker build -t $TRAIN --cache-from $TRAIN -f train.dockerfile $BUILD_ARGS .
docker build -t $PLAY --cache-from $PLAY -f play.dockerfile $BUILD_ARGS .