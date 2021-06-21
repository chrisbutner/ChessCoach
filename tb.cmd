pushd "%~dp0"
call activate_virtual_env.cmd
tensorboard --logdir %localappdata%\ChessCoach\TensorBoard
popd