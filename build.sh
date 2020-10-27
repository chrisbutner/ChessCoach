#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

if [[ $# -eq 0 ]] ; then
  echo ERROR: Expected 'debug' or 'release' build type argument
  exit 1
fi

case $1 in
  debug|release)
    BUILD_TYPE=$1
    shift
    ;;
  *)
    echo ERROR: Expected 'debug' or 'release' build type argument
    exit 1
    ;;
esac

BUILD_DIRECTORY=build/gcc/$BUILD_TYPE

if [ ! -f ${BUILD_DIRECTORY}/build.ninja ]; then
  meson setup $BUILD_DIRECTORY --buildtype $BUILD_TYPE
fi

ninja -C $BUILD_DIRECTORY "$@"