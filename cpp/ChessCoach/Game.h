#ifndef _GAME_H_
#define _GAME_H_

#include <map>
#include <cmath>
#include <algorithm>
#include <array>

#include <Stockfish/position.h>

#include "Network.h"

constexpr const static float CHESSCOACH_VALUE_WIN = 1.0f;
constexpr const static float CHESSCOACH_VALUE_DRAW = 0.5f;
constexpr const static float CHESSCOACH_VALUE_LOSS = 0.0f;
constexpr const static float CHESSCOACH_VALUE_UNINITIALIZED = -1.0f;

class Game
{
public:

    static void Initialize();

    constexpr static float CentipawnConversionPhase = 1.5620688421f;
    constexpr static float CentipawnConversionScale = 111.714640912f;

    inline static float CentipawnsToProbability(float centipawns)
    {
        // Use lc0 conversion for now (12800 = 1; for Stockfish 10k is found win, 32k is found mate).
        const float probability11 = (::atanf(centipawns / CentipawnConversionScale) / CentipawnConversionPhase);
        const float probability01 = INetwork::MapProbability11To01(probability11);
        return std::clamp(probability01, 0.f, 1.f);
    }

    inline static float ProbabilityToCentipawns(float probability01)
    {
        // Use lc0 conversion for now (12800 = 1; for Stockfish 10k is found win, 32k is found mate).
        const float probability11 = INetwork::MapProbability01To11(probability01);
        const float centipawns = (::tanf(probability11 * CentipawnConversionPhase) * CentipawnConversionScale);
        return centipawns;
    }

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

    constexpr static const int NoProgressSaturationCount = 99;
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

    static std::array<float, 25> UcbMateTerm;

public:

    Game();
    Game(const std::string& fen, const std::vector<Move>& moves);

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
    bool StockfishCanEvaluate() const;
    float StockfishEvaluation() const;

    inline Position& DebugPosition() { return _position; }
    inline Position& DebugPreviousPosition(int index) { return _previousPositions[index]; }

private:

    void GeneratePiecePlanes(INetwork::InputPlanes& image, int planeOffset, const Position& position) const;
    void FillPlane(INetwork::Plane& plane, float value) const;
    Key Rotate(Key key, unsigned int distance) const;

protected:

    // Used for both real and scratch games.
    Position _position;
    StateListPtr _positionStates;
    std::vector<Position> _previousPositions;
    int _previousPositionsOldest;
};

static_assert(CHESSCOACH_VALUE_WIN > CHESSCOACH_VALUE_DRAW);
static_assert(CHESSCOACH_VALUE_DRAW > CHESSCOACH_VALUE_LOSS);
static_assert(CHESSCOACH_VALUE_UNINITIALIZED == CHESSCOACH_VALUE_UNINITIALIZED);

static_assert(Game::FlipValue(CHESSCOACH_VALUE_WIN) == CHESSCOACH_VALUE_LOSS);
static_assert(Game::FlipValue(CHESSCOACH_VALUE_LOSS) == CHESSCOACH_VALUE_WIN);
static_assert(Game::FlipValue(CHESSCOACH_VALUE_DRAW) == CHESSCOACH_VALUE_DRAW);

static_assert(INetwork::MapProbability01To11(CHESSCOACH_VALUE_WIN) == NETWORK_VALUE_WIN);
static_assert(INetwork::MapProbability01To11(CHESSCOACH_VALUE_DRAW) == NETWORK_VALUE_DRAW);
static_assert(INetwork::MapProbability01To11(CHESSCOACH_VALUE_LOSS) == NETWORK_VALUE_LOSS);

static_assert(INetwork::MapProbability11To01(NETWORK_VALUE_WIN) == CHESSCOACH_VALUE_WIN);
static_assert(INetwork::MapProbability11To01(NETWORK_VALUE_DRAW) == CHESSCOACH_VALUE_DRAW);
static_assert(INetwork::MapProbability11To01(NETWORK_VALUE_LOSS) == CHESSCOACH_VALUE_LOSS);

#endif // _GAME_H_