ARG PROJECT_ID
ARG BASE_TAG
FROM gcr.io/${PROJECT_ID}/chesscoach-base:${BASE_TAG}
ARG NETWORK

# Playing machine should only "play".
# Re-install after modifying config.toml.
RUN sed -i -e "s/training_network_name.*=.*\".*\"/training_network_name = \"${NETWORK}\"/" \
  -e "s/uci_network_name.*=.*\".*\"/uci_network_name = \"${NETWORK}\"/" \
  -e "s/role.*=.*\".*\"/role = \"play\"/" \
  /chesscoach/config.toml && \
  /chesscoach/build.sh release install

CMD ["ChessCoachTrain"]
