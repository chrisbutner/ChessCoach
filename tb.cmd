pushd "%~dp0"
call conda activate chesscoach
tensorboard --logdir %localappdata%\ChessCoach\TensorBoard