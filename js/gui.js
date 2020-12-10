﻿(function () {

    function highlight(element, proportion) {
        if (proportion > 0) {
            proportion = 0.6 * proportion + 0.2;
        }
        element.style.backgroundColor = `rgba(128, 0, 0, ${proportion}`;
    }

    function overlaysHighlightAll() {
        const proportions = {}
        for (const square in overlays) {
            proportions[square] = 0;
        }
        for (const move of data.policy) {
            proportions[move.from] = Math.min(1, proportions[move.from] + move.value);
            proportions[move.to] = Math.min(1, proportions[move.to] + move.value);
        }
        for (const square in proportions) {
            highlight(overlays[square], proportions[square]);
        }
    }

    function overlaysHighlightMove(move) {
        for (const overlay of Object.values(overlays)) {
            overlay.style.backgroundColor = null;
        }
        highlight(overlays[move.from], move.value);
        highlight(overlays[move.to], move.value);
    }

    function clearMoves() {
        document.getElementById("moves").innerHTML = "";
    }

    function addMove(move) {
        const element = document.createElement("div");
        element.className = "move";
        document.getElementById("moves").appendChild(element);

        const san = document.createElement("span");
        san.className = "san";
        san.textContent = move.san;
        element.appendChild(san);

        const details = document.createElement("span");
        details.className = "details";
        details.textContent = `${move.value}`;
        element.appendChild(details);

        element.addEventListener("mouseover", () => {
            highlight(element, move.value);
            overlaysHighlightMove(move);
        });

        element.addEventListener("mouseout", () => {
            element.style.backgroundColor = null;
            overlaysHighlightAll();
        });
    }

    function requestPosition() {
        socket.send(JSON.stringify({
            type: "request",
            game: game,
            position: position,
        }));
    }

    function handleInitialize(message) {
        game_count = message.game_count

        if (game_count > 0) {
            requestPosition();
        }
    }

    function handleData(newData) {
        data = newData

        game = data.game;
        position_count = data.position_count;
        position = data.position;
        board.position(data.fen);

        displayGame = (game + 1).toString()
        displayPosition = (position + 1).toString()
        document.getElementById("gameInput").value = displayGame;
        document.getElementById("positionInput").value = displayPosition;
        document.getElementById("fenInfo").textContent = data.fen;
        document.getElementById("gameInfo").textContent = `Game ${displayGame} of ${game_count}`
        document.getElementById("positionInfo").textContent = `Position ${displayPosition} of ${position_count}`

        data.policy.sort((a, b) => { return b.value - a.value; });

        clearMoves();
        for (const move of data.policy) {
            addMove(move);
        }

        overlaysHighlightAll();
    }

    const config = {
        pieceTheme: "chessboardjs/img/chesspieces/wikipedia/{piece}.png",
        position: "start",
        draggable: true,
        dropOffBoard: "trash",
        sparePieces: true,
    };

    const board = Chessboard("board", config);
    const squares = {};
    const overlays = {};
    let game_count = 0;
    let game = 0;
    let position_count = 0;
    let position = 0;

    const sparePieces = document.getElementById("sparePieces");
    const sparePiecesTop = document.querySelector(".spare-pieces-top-4028b");
    const sparePiecesBottom = document.querySelector(".spare-pieces-bottom-ae20f");

    document.getElementById("board").appendChild(sparePieces);
    sparePieces.appendChild(sparePiecesBottom);
    sparePieces.appendChild(sparePiecesTop);

    document.getElementById("togglePieces").addEventListener("click", () => {
        sparePieces.style.display = (getComputedStyle(sparePieces, null).display === "none") ? "block" : "none";
    });

    document.getElementById("startButton").addEventListener("click", () => {
        board.start(false);
    });

    document.getElementById("clearButton").addEventListener("click", () => {
        board.clear(false);
    });

    for (const file of "abcdefgh") {
        for (let rank = 1; rank <= 8; rank++) {
            const square = file + rank;
            const squareElement = document.querySelector(".square-" + square);
            const squareOverlay = document.createElement("div");
            squareOverlay.style.width = getComputedStyle(squareElement, null).width;
            squareOverlay.style.height = getComputedStyle(squareElement, null).height;
            squareOverlay.style.position = "absolute";
            squareOverlay.style.top = "0px";
            squareOverlay.style.left = "0px";
            squareElement.appendChild(squareOverlay);

            squares[square] = squareElement;
            overlays[square] = squareOverlay;
        }
    }

    messageHandlers = {
        "ping": () => { },
        "initialize": handleInitialize,
        "data": handleData,
    };

    const socket = new WebSocket("ws://localhost:8001/");
    socket.addEventListener("message", (e) => {
        message = JSON.parse(e.data);
        messageHandlers[message.type](message);
    });
    socket.addEventListener("open", (e) => {
        socket.send(JSON.stringify({ type: "hello" }));
    });

    document.getElementById("gamePrevBig").addEventListener("click", (e) => {
        if (game_count > 0) {
            game -= 2000;
            position = null;
            requestPosition();
        }
    });
    document.getElementById("gamePrev").addEventListener("click", (e) => {
        if (game_count > 0) {
            game--;
            position = null;
            requestPosition();
        }
    });
    document.getElementById("gameNext").addEventListener("click", (e) => {
        if (game_count > 0) {
            game++;
            position = null;
            requestPosition();
        }
    });
    document.getElementById("gameNextBig").addEventListener("click", (e) => {
        if (game_count > 0) {
            game += 2000;
            position = null;
            requestPosition();
        }
    });

    document.getElementById("positionPrevBig").addEventListener("click", (e) => {
        if (game_count > 0) {
            // Don't accidentally wrap to negative: -1 means last.
            position = Math.max(0, position - 10);
            requestPosition();
        }
    });
    document.getElementById("positionPrev").addEventListener("click", (e) => {
        if (game_count > 0) {
            // Don't accidentally wrap to negative: -1 means last.
            position = Math.max(0, position - 1);
            requestPosition();
        }
    });
    document.getElementById("positionNext").addEventListener("click", (e) => {
        if (game_count > 0) {
            position++;
            requestPosition();
        }
    });
    document.getElementById("positionNextBig").addEventListener("click", (e) => {
        if (game_count > 0) {
            position += 10;
            requestPosition();
        }
    });

    gameInput = document.getElementById("gameInput");
    gameInput.addEventListener("focus", (e) => {
        gameInput.select();
    });
    gameInput.addEventListener("blur", (e) => {
        if (game_count > 0) {
            value = (parseInt(gameInput.value) - 1);
            if (!isNaN(value) && (value !== game)) {
                game = value;
                position = null;
                requestPosition();
            } else {
                gameInput.value = (game + 1).toString();
            }
        }
    });
    positionInput = document.getElementById("positionInput");
    positionInput.addEventListener("focus", (e) => {
        positionInput.select();
    });
    positionInput.addEventListener("blur", (e) => {
        if (game_count > 0) {
            value = (parseInt(positionInput.value) - 1);
            if (!isNaN(value) && (value !== position)) {
                position = value;
                requestPosition();
            } else {
                positionInput.value = (position + 1).toString();
            }
        }
    });
})();