#!/usr/bin/env bash
set -eux
pushd "$(dirname "$0")"

# Generate .cpp/.h from .proto
protoc -I=cpp/protobuf --cpp_out=cpp/protobuf ChessCoach.proto