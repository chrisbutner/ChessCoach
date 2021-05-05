#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

# Install required build packages and dependencies
sudo apt-get update
# First line our dependencies, second line protobuf dependencies
sudo apt-get install -y --no-install-recommends \
  meson g++ pkg-config python3-dev libgtest-dev zlib1g-dev python3-pip \
  autoconf automake libtool curl make g++ unzip
# Our dependencies
pip3 install toml scikit-optimize matplotlib

# Install cutechess dependencies
sudo apt-get install -y --no-install-recommends qt5-default

# Build and install protobuf (leaving the build directory around for flexibility).
curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz | tar -xz
pushd protobuf-3.13.0
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
popd

popd