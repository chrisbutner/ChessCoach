#ifndef _GAME_H_
#define _GAME_H_

#include <map>
#include <cmath>
#include <algorithm>
#include <array>

#include <Stockfish/position.h>

#include "Network.h"
#include "PoolAllocator.h"

constexpr const static float CHESSCOACH_VALUE_WIN = 1.0f;
constexpr const static float CHESSCOACH_VALUE_DRAW = 0.5f;
constexpr const static float CHESSCOACH_VALUE_LOSS = 0.0f;
constexpr const static float CHESSCOACH_VALUE_UNINITIALIZED = -1.0f;

constexpr const static float CHESSCOACH_FIRST_PLAY_URGENCY = CHESSCOACH_VALUE_LOSS;

class Game
{
public:

    static const size_t BlockSizeBytes = 64 * 1024 * 1024; // 64 MiB
    thread_local static PoolAllocator<StateInfo, BlockSizeBytes> StateAllocator;

    static StateInfo* AllocateState();
    static void FreeState(StateInfo* state);

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

    static Bitboard FlipBoard(Bitboard board)
    {
#ifdef CHESSCOACH_WINDOWS
        return _byteswap_uint64(board);
#else
        return __builtin_bswap64(board);
#endif
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

    constexpr const static INetwork::PackedPlane FillPlanePacked[COLOR_NB] = { 0, static_cast<INetwork::PackedPlane>(0xFFFFFFFFFFFFFFFFULL) };

    const static int NO_PLANE = -1;

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
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    };

    static std::array<float, 25> PuctMateTerm;

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
    void ApplyMoveMaybeNull(Move move);
    Move ApplyMoveInfer(const INetwork::PackedPlane* resultingPieces);
    Move ApplyMoveGuess(float result, const std::map<Move, float>& policy);
    bool IsDrawByNoProgressOrThreefoldRepetition() const;
    bool PiecesMatch(const INetwork::PackedPlane* a, const INetwork::PackedPlane* b) const;
    int Ply() const;
    Key GenerateImageKey() const;
    void GenerateImage(INetwork::InputPlanes& imageOut) const;
    void GenerateImage(INetwork::PackedPlane* imageOut) const;
    void GenerateImageCompressed(INetwork::PackedPlane* piecesOut, INetwork::PackedPlane* auxiliaryOut) const;
    float& PolicyValue(INetwork::OutputPlanes& policy, Move move) const;
    float& PolicyValue(INetwork::PlanesPointerFlat policyInOut, Move move) const;
    float& PolicyValue(INetwork::PlanesPointer policyInOut, Move move) const;
    void GeneratePolicy(const std::map<Move, float>& childVisits, INetwork::OutputPlanes& policyOut) const;
    void GeneratePolicyCompressed(const std::map<Move, float>& childVisits, int64_t* policyIndicesOut, float* policyValuesOut) const;
    void GeneratePolicyDecompress(int childVisitsSize, const int64_t* policyIndices, const float* policyValues, INetwork::OutputPlanes& policyOut);
    bool StockfishCanEvaluate() const;
    float StockfishEvaluation() const;
    const Position& GetPosition() const;

    inline Position& DebugPosition() { return _position; }
    inline Position& DebugPreviousPosition(int index) { return _previousPositions[index]; }

private:

    void Free();
    void GeneratePiecePlanes(INetwork::PackedPlane* imageOut, int planeOffset, const Position& position) const;
    void FillPlane(INetwork::PackedPlane& plane, bool value) const;
    Key Rotate(Key key, unsigned int distance) const;

protected:

    // Used for both real and scratch games.
    Position _position;
    StateInfo* _parentState;
    StateInfo* _currentState;
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

static_assert(Game::FlipValue(CHESSCOACH_VALUE_UNINITIALIZED) != CHESSCOACH_VALUE_UNINITIALIZED);
static_assert(Game::FlipValue(CHESSCOACH_VALUE_UNINITIALIZED) != CHESSCOACH_VALUE_WIN);
static_assert(Game::FlipValue(CHESSCOACH_VALUE_UNINITIALIZED) != NETWORK_VALUE_DRAW);
static_assert(Game::FlipValue(CHESSCOACH_VALUE_UNINITIALIZED) != NETWORK_VALUE_LOSS);

#endif // _GAME_H_