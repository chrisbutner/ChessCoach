#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

# Install required build packages and dependencies.
sudo apt-get update
# First line our dependencies, second line protobuf dependencies
sudo apt-get install -y --no-install-recommends \
  meson g++ pkg-config python3-dev libgtest-dev zlib1g-dev python3-pip \
  autoconf automake libtool curl make g++ unzip

# Install gsutil.
sudo apt-get install -y --no-install-recommends curl gnupg && \
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
  sudo apt-get update -y && \
  sudo apt-get install -y --no-install-recommends google-cloud-sdk

# Install our Python dependencies. Don't install TensorFlow yet, as it varies between alpha and non-alpha TPUs.
pip3 install -U pip
pip3 install setuptools
pip3 install -r requirements-linux-common.txt

# Install cutechess dependencies (Ubuntu 20.04 || Ubuntu 22.04).
sudo bash -c 'apt-get install -y --no-install-recommends qt5-default || apt-get install -y --no-install-recommends qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools'

# Build and install protobuf.
curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz | tar -xz
pushd protobuf-3.13.0
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
popd

# Clean up after installing protobuf.
rm -r --interactive=never protobuf-3.13.0

popd