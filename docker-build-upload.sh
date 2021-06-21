#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

source docker-build.sh

docker push $BASE
docker push $TRAIN
docker push $PLAY
docker push $PROXY

popd