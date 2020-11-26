FROM debian:10

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

RUN pip3 install --upgrade pip && \
  pip3 install setuptools && \
  pip3 install toml tensorflow-cpu==2.3.1 cloud-tpu-client

COPY . /chesscoach
RUN /chesscoach/build.sh release install && \
  rm -r /chesscoach/build