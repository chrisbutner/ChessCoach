call conda activate chesscoach
pyinstaller ^
	--clean ^
	--noconfirm ^
	--name "chesscoach" ^
	chesscoach.spec