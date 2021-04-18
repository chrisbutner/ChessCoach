#!/usr/bin/env bash
set -eux

pip3 install --user meson
sudo apt-get install -y --no-install-recommends g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8

# sudo "PATH=~/.local/bin:$PATH" ./build.sh release install