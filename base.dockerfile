FROM ubuntu:20.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends meson g++ pkg-config python3-dev libgtest-dev zlib1g-dev python3-pip autoconf automake libtool curl make g++ unzip

RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz | tar -xz && \
  cd protobuf-3.13.0 && \
  ./configure && \
  make && \
  make check && \
  make install && \
  ldconfig && \
  cd .. && \
  rm -r protobuf-3.13.0

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# RUN pip3 install --upgrade pip && \
#   pip3 install setuptools && \
#   pip3 install toml tensorflow-cpu==2.4.0 cloud-tpu-client

RUN pip3 install setuptools toml

COPY . /chesscoach
RUN /chesscoach/build.sh release install && \
  rm -r /chesscoach/build

ENV PYTHONUNBUFFERED=1