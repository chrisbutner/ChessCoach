#ifndef _SELFPLAY_H_
#define _SELFPLAY_H_

#include <unordered_map>
#include <vector>
#include <random>

#include <Stockfish/Position.h>
#include <Stockfish/movegen.h>

#include "Network.h"

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

    Game();

    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&& other) = delete;
    Game& operator=(Game && other) = delete;
    ~Game();

    Node* Root() const;
    bool IsTerminal() const;
    Color ToPlay() const;
    void ApplyMove(Move move, Node* newRoot);
    float ExpandAndEvaluate(const INetwork* network);
    std::vector<float> Softmax(const std::vector<float>& logits) const;
    float GetLogit(const OutputPlanesPtr policy, Move move) const;
    InputPlanes MakeImage() const;
    std::pair<Move, Node*> SelectMove() const;
    void StoreSearchStatistics();
    int Ply() const;

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

    void Play() const;
    std::pair<Move, Node*> RunMcts(const INetwork* network, Game& game) const;
    void AddExplorationNoise(Game& game) const;
    std::pair<Move, Node*> SelectChild(const Node* node) const;
    float CalculateUcbScore(const Node* parent, const Node* child) const;
    void Backpropagate(const std::vector<Node*>& searchPath, float value) const;
    float FlipValue(Color toPlay, float value) const;
    float FlipValue(float value) const;
    void Prune(Node* root, Node* except) const;
    void PruneAll(Node* root) const;
};

#endif // _SELFPLAY_H_