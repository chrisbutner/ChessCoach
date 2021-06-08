ARG PROJECT_ID
ARG BASE_TAG
FROM eu.gcr.io/${PROJECT_ID}/chesscoach-base:${BASE_TAG}
ARG NETWORK

# Playing machine should only "play".
RUN sed -i -e "s/^network_name.*=.*\".*\"/network_name = \"${NETWORK}\"/" \
  -e "s/role.*=.*\".*\"/role = \"play\"/" \
  /usr/local/share/ChessCoach/config.toml

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# CMD ["ChessCoachTrain"]
CMD pip3 install wheel && \
  pip3 install /usr/share/tpu/tf_nightly*.whl && \
  exec ChessCoachTrain
