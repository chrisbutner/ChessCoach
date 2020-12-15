#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

source cluster-common.sh
source cluster-prep-creds.sh

# Taint the default node pool so that critical system pods like "kube-dns" are instead scheduled on
# our separate critical node pool, which isn't preemptible.
gcloud container clusters create \
    --cluster-version=${CLUSTER_VERSION} \
    --scopes=cloud-platform \
    --enable-ip-alias \
    --enable-tpu \
    --machine-type ${MACHINE_TYPE} \
    --disk-size ${DISK_SIZE} \
    --preemptible \
    --num-nodes ${NUM_NODES} \
    --node-taints=dedicated=preemptible:NoSchedule \
    ${CLUSTER_NAME}

gcloud container node-pools create \
    --cluster=${CLUSTER_NAME} \
    --scopes=cloud-platform \
    --machine-type ${CRITICAL_MACHINE_TYPE} \
    --disk-size ${CRITICAL_DISK_SIZE} \
    --num-nodes ${CRITICAL_NUM_NODES} \
    ${CRITICAL_NAME}

kubectl create secret generic ${CREDENTIALS_NAME} --from-file=key.json=${SERVICE_ACCOUNT_KEY_PATH}