pushd "%~dp0"

set CHESSCOACH_DATA=%localappdata%\ChessCoach
set CHESSCOACH_DATA_ZIP="%CHESSCOACH_DATA%\download_install_data.zip"

mkdir "%CHESSCOACH_DATA%"

curl -L https://github.com/chrisbutner/ChessCoachData/releases/download/v1.0.0/Data.zip -o "%CHESSCOACH_DATA_ZIP%"

tar -x -f "%CHESSCOACH_DATA_ZIP%" -C "%CHESSCOACH_DATA%"

del "%CHESSCOACH_DATA_ZIP%"

popd