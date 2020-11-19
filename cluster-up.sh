#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

source cluster-common.sh
source cluster-prep-creds.sh

gcloud container clusters create \
    --cluster-version=${CLUSTER_VERSION} \
    --scopes=cloud-platform \
    --enable-ip-alias \
    --enable-tpu \
    --machine-type ${MACHINE_TYPE} \
    --disk-size ${DISK_SIZE} \
    --preemptible \
    --num-nodes ${NUM_NODES} \
    ${CLUSTER_NAME}

kubectl create secret generic ${CREDENTIALS_NAME} --from-file=key.json=${SERVICE_ACCOUNT_KEY_PATH}