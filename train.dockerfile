ARG PROJECT_ID
ARG BASE_TAG
FROM gcr.io/${PROJECT_ID}/chesscoach-base:${BASE_TAG}
ARG NETWORK

# Training machine should really "train|play".
# Re-install after modifying config.toml.
RUN sed -i -e "s/training_network_name.*=.*\".*\"/training_network_name = \"${NETWORK}\"/" \
  -e "s/uci_network_name.*=.*\".*\"/uci_network_name = \"${NETWORK}\"/" \
  -e "s/role.*=.*\".*\"/role = \"train|play\"/" \
  /chesscoach/config.toml && \
  /chesscoach/build.sh release install && \
  rm -r /chesscoach/build

CMD ["ChessCoachTrain"]
