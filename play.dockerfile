ARG PROJECT_ID
ARG BASE_TAG
FROM gcr.io/${PROJECT_ID}/chesscoach-base:${BASE_TAG}
ARG NETWORK

# Playing machine should only "play".
# Re-install after modifying config.toml.
# Ubuntu 18.04: include user-installed meson in PATH.
RUN sed -i -e "s/training_network_name.*=.*\".*\"/training_network_name = \"${NETWORK}\"/" \
  -e "s/uci_network_name.*=.*\".*\"/uci_network_name = \"${NETWORK}\"/" \
  -e "s/role.*=.*\".*\"/role = \"play\"/" \
  /chesscoach/config.toml && \
  PATH=~/.local/bin:$PATH /chesscoach/build.sh release install && \
  rm -r /chesscoach/build

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# Also, gs:// paths are currently bugged, so set up gcsfuse as a workaround.
# CMD ["ChessCoachTrain"]
CMD pip3 install wheel && \
  pip3 install /usr/share/tpu/tf_nightly*.whl && \
  gcsfuse --implicit-dirs chesscoach /tmp/gcs/chesscoach && \
  ChessCoachTrain
