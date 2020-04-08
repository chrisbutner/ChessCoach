#ifndef _GAME_H_
#define _GAME_H_

#include <map>

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

    constexpr static Move FlipMove(Color color, Move move)
    {
        return Move(static_cast<int>(move) ^ FlipMoveMask[color]);
    }

    constexpr static Square FlipSquare(Color color, Square square)
    {
        return Square(static_cast<int>(square) ^ FlipSquareMask[color]);
    }

    constexpr static float FlipValue(Color toPlay, float value)
    {
        return (toPlay == WHITE) ? value : FlipValue(value);
    }

    constexpr static float FlipValue(float value)
    {
        return (CHESSCOACH_VALUE_WIN - value);
    }

    constexpr static Piece FlipPiece[COLOR_NB][PIECE_NB] =
    {
        { NO_PIECE, W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, NO_PIECE,
            NO_PIECE, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE },
        { NO_PIECE, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE,
            NO_PIECE, W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, NO_PIECE },
    };

    constexpr const static int FlipMoveMask[COLOR_NB] = { 0, ((SQ_A8 << 6) + static_cast<int>(SQ_A8)) };
    constexpr const static int FlipSquareMask[COLOR_NB] = { 0, SQ_A8 };

    constexpr const static CastlingRights KingsideRights[COLOR_NB] = { WHITE_OO, BLACK_OO };
    constexpr const static CastlingRights QueensideRights[COLOR_NB] = { WHITE_OOO, BLACK_OOO };

    // TODO: Later optimization idea to benchmark: could allocate one extra plane,
    // let the -1s go in without branching, then pass [1] reinterpreted to consumers
    const static int NO_PLANE = -1;
    const static int InputPiecePlaneCount = 12;

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

    static const int NoProgressSaturationCount = 99;

    static Key PredictionCache_PreviousMoveSquare[INetwork::InputPreviousMoveCount][SQUARE_NB];
    static Key PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];

    constexpr const static char* SquareName[SQUARE_NB] = {
        "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
        "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
        "A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
        "A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
        "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
        "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
        "A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
        "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8",
    };

    

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
    float& PolicyValue(INetwork::OutputPlanes& policy, Move move) const;
    Key GenerateImageKey() const;
    INetwork::InputPlanes GenerateImage() const;
    INetwork::OutputPlanes GeneratePolicy(const std::map<Move, float>& childVisits) const;

    inline Position& DebugPosition() { return _position; }

private:

    void FillPlane(INetwork::Plane& plane, float value) const;

protected:

    // Used for both real and scratch games.
    Position _position;
    StateListPtr _positionStates;
    std::array<Move, INetwork::InputPreviousMoveCount> _previousMoves;
    int _previousMovesOldest;
};

#endif // _GAME_H_