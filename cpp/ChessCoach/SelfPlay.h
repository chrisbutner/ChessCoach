#ifndef _SELFPLAY_H_
#define _SELFPLAY_H_

#include <unordered_map>
#include <vector>
#include <random>

#include <Stockfish/Position.h>
#include <Stockfish/movegen.h>

#include "Network.h"
#include "Storage.h"

const static float CHESSCOACH_VALUE_WIN = 1.0f;
const static float CHESSCOACH_VALUE_DRAW = 0.5f;
const static float CHESSCOACH_VALUE_LOSE = 0.0f;
const static float CHESSCOACH_VALUE_UNINITIALIZED = -1.0f;

struct Config
{
    static int NumSampingMoves;
    static int MaxMoves;
    static int NumSimulations;

    static float RootDirichletAlpha;
    static float RootExplorationFraction;

    static float PbCBase;
    static float PbCInit;

    // TODO: Move this if needed for threading, etc.
    static std::default_random_engine Random;
};

class Node
{

public:

    Node(float setPrior);

    bool IsExpanded() const;
    float Value() const;
    int SumChildVisits() const;

    std::unordered_map<Move, Node*> children;
    float originalPrior;
    float prior;
    int visitCount;
    float valueSum;
    float terminalValue;

private:

    // Doesn't strictly follow "mutable" if SumChildVisits() is misused and
    // called before children are changed while this node is still around.
    mutable int _sumChildVisits;
};

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

public:

    Game();

    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&& other) = delete;
    Game& operator=(Game && other) = delete;
    ~Game();

    Node* Root() const;
    bool IsTerminal() const;
    float TerminalValue() const;
    Color ToPlay() const;
    void ApplyMove(Move move, Node* newRoot);
    void ApplyMoveWithHistory(Move move, Node* newRoot);
    float ExpandAndEvaluate(INetwork* network);
    std::vector<float> Softmax(const std::vector<float>& logits) const;
    std::pair<Move, Node*> SelectMove() const;
    void StoreSearchStatistics();
    int Ply() const;
    StoredGame Store();
    float& PolicyValue(OutputPlanesPtr policy, Move move) const;
    InputPlanes GenerateImage() const;
    OutputPlanes GeneratePolicy(const std::unordered_map<Move, float>& childVisits) const;

private:

    // Used for both real and scratch games.
    Position _position;
    StateListPtr _positionStates;
    Node* _root;

    // Only used for real games, so no need to copy.
    std::vector<std::unordered_map<Move, float>> _childVisits;
    std::vector<Move> _history;
};

class Mcts
{
public:

    void Work(INetwork* network) const;
    void Play(INetwork* network) const;
    std::pair<Move, Node*> RunMcts(INetwork* network, Game& game) const;
    void AddExplorationNoise(Game& game) const;
    std::pair<Move, Node*> SelectChild(const Node* node) const;
    float CalculateUcbScore(const Node* parent, const Node* child) const;
    void Backpropagate(const std::vector<Node*>& searchPath, float value) const;
    float FlipValue(Color toPlay, float value) const;
    float FlipValue(float value) const;
    void Prune(Node* root, Node* except) const;
    void PruneAll(Node* root) const;

private:

    Storage _storage;
};

#endif // _SELFPLAY_H_