#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

kubectl create -f cluster-train-deployment.yaml
kubectl create -f cluster-play-deployment.yaml