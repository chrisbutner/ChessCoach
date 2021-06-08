ARG PROJECT_ID
ARG BASE_TAG
FROM eu.gcr.io/${PROJECT_ID}/chesscoach-base:${BASE_TAG}
ARG NETWORK

# With 1-25 v3-8 TPUs it's safe to use "train|play" for the trainer,
# so that it can play games in between training.
#
# With more TPUs, too little time is available for playing, so games
# end up stretching over too many older networks: just use "train".
RUN sed -i -e "s/^network_name.*=.*\".*\"/network_name = \"${NETWORK}\"/" \
  -e "s/role.*=.*\".*\"/role = \"train\"/" \
  /usr/local/share/ChessCoach/config.toml

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# CMD ["ChessCoachTrain"]
CMD pip3 install wheel && \
  pip3 install /usr/share/tpu/tf_nightly*.whl && \
  exec ChessCoachTrain
