#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

# Install required build packages and dependencies
sudo apt install meson g++ pkg-config python3-dev libgtest-dev zlib1g-dev	# our dependencies
sudo apt install autoconf automake libtool curl make g++ unzip				# protobuf build dependencies
sudo pip3 install toml														# our dependencies

# Build and install protobuf
mkdir temp
cd temp
curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz | tar -xz
cd protobuf-3.13.0
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.