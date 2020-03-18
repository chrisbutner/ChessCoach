#include <iostream>

#include "bitboard.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "movegen.h"

int main(int argc, char* argv[])
{
    Bitboards::init();
    Position::init();
    Bitbases::init();

    // Initialize threads. They directly reach out to UCI options, so we need to initialize that too.
    UCI::init(Options);
    Threads.set(Options["Threads"]);

    // Set up the starting position.
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    StateListPtr states(new std::deque<StateInfo>(1));
    Position position;
    position.set(StartFEN, false /* isChess960 */, &states->back(), Threads.main());  

    // Generate legal moves.
    ExtMove moves[MAX_MOVES];
    ExtMove* cur = moves;
    ExtMove* endMoves = generate<LEGAL>(position, cur);

    // Debug
    std::cout << "Legal moves: " << (endMoves - cur) << std::endl;
    while (cur != endMoves)
    {
        std::cout << from_sq(cur->move) << " to " << to_sq(cur->move) << std::endl;
        cur++;
    }

    // Clean up.
    Threads.set(0);
    return 0;
}
