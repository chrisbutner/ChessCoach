(function () {
    let config = {
        pieceTheme: "chessboardjs/img/chesspieces/wikipedia/{piece}.png",
        position: "start",
        draggable: true,
        dropOffBoard: "trash",
        sparePieces: true,
    };

    let board = Chessboard("board", config);

    let sparePieces = document.getElementById("sparePieces");
    let sparePiecesTop = document.querySelector(".spare-pieces-top-4028b");
    let sparePiecesBottom = document.querySelector(".spare-pieces-bottom-ae20f");

    document.getElementById("board").appendChild(sparePieces);
    sparePieces.appendChild(sparePiecesBottom);
    sparePieces.appendChild(sparePiecesTop);

    document.getElementById("togglePieces").addEventListener("click", function () {
        sparePieces.style.display = (getComputedStyle(sparePieces, null).display === "none") ? "block" : "none";
    });

    document.getElementById("startButton").addEventListener("click", function () {
        board.start(false);
    });

    document.getElementById("clearButton").addEventListener("click", function () {
        board.clear(false);
    });
})();