#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

source cluster-common.sh

# Create the service account if it doesn't exist.
if ! gcloud iam service-accounts list | grep -q ${SERVICE_ACCOUNT_NAME}; then
  gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME
fi

# Parse the email.
SERVICE_ACCOUNT_EMAIL=$(gcloud iam service-accounts list | grep $SERVICE_ACCOUNT_NAME | grep -oP '[^\s]+@[^\s]+')

# Create a new key if not found locally.
if [ ! -f $SERVICE_ACCOUNT_KEY_PATH ]; then
  gcloud iam service-accounts keys create $SERVICE_ACCOUNT_KEY_PATH --iam-account $SERVICE_ACCOUNT_EMAIL
fi

# Grant the "storage.admin" role to the service account, matching the way things are set up for TPU workers.
gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$SERVICE_ACCOUNT_EMAIL --role=roles/storage.admin
