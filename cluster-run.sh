#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

kubectl apply -f cluster-train-deployment.yaml
kubectl apply -f cluster-play-deployment.yaml