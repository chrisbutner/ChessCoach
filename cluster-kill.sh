#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

kubectl delete -f cluster-train-deployment.yaml
kubectl delete -f cluster-play-deployment.yaml

popd