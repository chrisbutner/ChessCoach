#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

source cluster-common.sh

gcloud container clusters delete ${CLUSTER_NAME}