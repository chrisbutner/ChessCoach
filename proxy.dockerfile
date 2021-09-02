ARG PROJECT_ID
ARG BASE_TAG
ARG NETWORK
FROM eu.gcr.io/${PROJECT_ID}/chesscoach-play:${NETWORK}_${BASE_TAG}

# Google Cloud TPU VM Alpha: need custom TensorFlow wheel at runtime.
# CMD ["python3 /usr/local/bin/ChessCoach/uci_proxy_server.py ChessCoachUci"]
#
# Also, run network.py first so that Syzygy is replicated locally for both ChessCoach and Stockfish.
CMD pip3 install wheel && \
  pip3 install /usr/share/tpu/tf_nightly*.whl && \
  python3 /usr/local/bin/ChessCoach/network.py && \
  python3 /usr/local/bin/ChessCoach/uci_proxy_server.py ChessCoachUci 24377 & \
  python3 /usr/local/bin/ChessCoach/uci_proxy_server.py stockfish_13_linux_x64_bmi2 24378
