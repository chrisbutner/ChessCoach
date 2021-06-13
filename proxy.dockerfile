ARG PROJECT_ID
ARG BASE_TAG
ARG NETWORK
FROM eu.gcr.io/${PROJECT_ID}/chesscoach-play:${NETWORK}_${BASE_TAG}

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# CMD ["python3 /usr/local/bin/ChessCoach/uci_proxy_server.py ChessCoachUci"]
CMD pip3 install wheel && \
  pip3 install /usr/share/tpu/tf_nightly*.whl && \
  exec python3 /usr/local/bin/ChessCoach/uci_proxy_server.py ChessCoachUci
