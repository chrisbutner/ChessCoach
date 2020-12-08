(function () {

    function highlight(element, proportion) {
        element.style.backgroundColor = `rgba(128, 0, 0, ${proportion}`;
    }

    function overlaysHighlightAll() {
        const proportions = {}
        for (const square in overlays) {
            proportions[square] = 0;
        }
        for (const move of data.policy) {
            proportions[move.from] = Math.min(1, proportions[move.from] + move.proportion);
            proportions[move.to] = Math.min(1, proportions[move.to] + move.proportion);
        }
        for (const square in proportions) {
            highlight(overlays[square], proportions[square]);
        }
    }

    function overlaysHighlightMove(move) {
        for (const overlay of Object.values(overlays)) {
            overlay.style.backgroundColor = null;
        }
        highlight(overlays[move.from], move.proportion);
        highlight(overlays[move.to], move.proportion);
    }

    function clearMoves() {
        document.getElementById("moves").innerHTML = "";
    }

    function addMove(summary, move) {
        move.proportion = (move.visit_count / summary.visit_count);

        const element = document.createElement("div");
        element.className = "move";
        document.getElementById("moves").appendChild(element);

        const san = document.createElement("span");
        san.className = "san";
        san.textContent = move.san;
        element.appendChild(san);

        const details = document.createElement("span");
        details.className = "details";
        details.textContent = `${move.visit_count} / ${summary.visit_count} (${move.proportion})`;
        element.appendChild(details);

        element.addEventListener("mouseover", () => {
            highlight(element, move.proportion);
            overlaysHighlightMove(move);
        });

        element.addEventListener("mouseout", () => {
            element.style.backgroundColor = null;
            overlaysHighlightAll();
        });
    }

    function showData(newData) {
        data = newData;
        data.summary = { visit_count: data.policy.reduce((total, move) => total + move.visit_count, 0) };

        board.position(data.position);
        document.getElementById("position").textContent = data.position;

        clearMoves();
        for (const move of data.policy) {
            addMove(data.summary, move);
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
    let data = { position: "", value: -1, policy: [], summary: { visit_count: 1 } };

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

    const socket = new WebSocket("ws://localhost:8001/");
    socket.addEventListener("message", (e) => {
        if (e.data === "ping") {
            return;
        }
        showData(JSON.parse(e.data));
    });
})();