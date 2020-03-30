#ifndef _GAME_H_
#define _GAME_H_

#include <unordered_map>

#include <Stockfish/Position.h>

#include "Network.h"

const static float CHESSCOACH_VALUE_WIN = 1.0f;
const static float CHESSCOACH_VALUE_DRAW = 0.5f;
const static float CHESSCOACH_VALUE_LOSE = 0.0f;
const static float CHESSCOACH_VALUE_UNINITIALIZED = -1.0f;

class Game
{
public:

    static void Initialize();

    constexpr static Piece FlipPiece[COLOR_NB][PIECE_NB] =
    {
        { NO_PIECE, W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, NO_PIECE,
            NO_PIECE, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE },
        { NO_PIECE, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE,
            NO_PIECE, W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, NO_PIECE },
    };

    constexpr static Square RotateSquare(Color color, Square square)
    {
        return Square(square + color * (SQUARE_NB - 1 - 2 * square));
    }

    // TODO: Later optimization idea to benchmark: could allocate one extra plane,
    // let the -1s go in without branching, then pass [1] reinterpreted to consumers
    const static int NO_PLANE = -1;

    constexpr static int ImagePiecePlane[PIECE_NB] =
    {
        NO_PLANE, 0, 1, 2, 3, 4, 5, NO_PLANE,
        NO_PLANE, 6, 7, 8, 9, 10, 11, NO_PLANE,
    };

    // UnderpromotionPlane[Piece - KNIGHT][to - from - NORTH_WEST]
    constexpr static int UnderpromotionPlane[3][3] =
    {
        { 64, 65, 66, },
        { 67, 68, 69, },
        { 70, 71, 72, },
    };

    // QueenKnightPlane[(to - from + SQUARE_NB) % SQUARE_NB]
    static int QueenKnightPlane[SQUARE_NB];

    static float FlipValue(Color toPlay, float value);
    static float FlipValue(float value);

public:

    Game();

    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&& other) noexcept;
    Game& operator=(Game && other) noexcept;
    ~Game();

    Color ToPlay() const;
    void ApplyMove(Move move);
    int Ply() const;
    float& PolicyValue(OutputPlanes& policy, Move move) const;
    InputPlanes GenerateImage() const;
    OutputPlanes GeneratePolicy(const std::unordered_map<Move, float>& childVisits) const;

protected:

    // Used for both real and scratch games.
    Position _position;
    StateListPtr _positionStates;
};

#endif // _GAME_H_