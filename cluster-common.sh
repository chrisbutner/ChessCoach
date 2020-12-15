#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

SERVICE_ACCOUNT_NAME=chesscoach-cluster-account
SERVICE_ACCOUNT_KEY_PATH=key.json
CREDENTIALS_NAME=chesscoach-credentials

CLUSTER_VERSION=1.17
CLUSTER_NAME=chesscoach-cluster
NUM_NODES=4
MACHINE_TYPE=custom-4-36864-ext
DISK_SIZE=40

CRITICAL_NAME=critical-node-pool
CRITICAL_NUM_NODES=1
CRITICAL_MACHINE_TYPE=n1-standard-1
CRITICAL_DISK_SIZE=20