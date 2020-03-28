#ifndef _SELFPLAY_H_
#define _SELFPLAY_H_

#include <unordered_map>
#include <vector>
#include <random>
#include <atomic>

#include <Stockfish/Position.h>
#include <Stockfish/movegen.h>

#include "Game.h"
#include "Network.h"
#include "Storage.h"
#include "Threading.h"

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

class SelfPlayGame : public Game
{
public:

    static std::atomic_uint ThreadSeed;
    thread_local static std::default_random_engine Random;

public:

    SelfPlayGame();

    SelfPlayGame(const SelfPlayGame& other);
    SelfPlayGame& operator=(const SelfPlayGame& other);
    SelfPlayGame(SelfPlayGame&& other) = delete;
    SelfPlayGame& operator=(SelfPlayGame && other) = delete;
    ~SelfPlayGame();

    Node* Root() const;
    bool IsTerminal() const;
    float TerminalValue() const;
    float Result() const;
    
    void ApplyMoveWithRoot(Move move, Node* newRoot);
    void ApplyMoveWithRootAndHistory(Move move, Node* newRoot);
    float ExpandAndEvaluate(INetwork* network);
    std::vector<float> Softmax(const std::vector<float>& logits) const;
    std::pair<Move, Node*> SelectMove() const;
    void StoreSearchStatistics();
    StoredGame Store();

private:

    // Used for both real and scratch games.
    Node* _root;

    // Only used for real games, so no need to copy.
    std::vector<std::unordered_map<Move, float>> _childVisits;
    std::vector<Move> _history;
};

class Mcts
{
public:

    Mcts(Storage* storage);

    void PlayGames(WorkCoordinator& workCoordinator, INetwork* network) const;
    void TrainNetwork(INetwork* network, int stepCount, int checkpoint) const;
    bool Play(INetwork* network) const;
    std::pair<Move, Node*> RunMcts(INetwork* network, SelfPlayGame& game) const;
    void AddExplorationNoise(SelfPlayGame& game) const;
    std::pair<Move, Node*> SelectChild(const Node* node) const;
    float CalculateUcbScore(const Node* parent, const Node* child) const;
    void Backpropagate(const std::vector<Node*>& searchPath, float value) const;
    void Prune(Node* root, Node* except) const;
    void PruneAll(Node* root) const;

private:

    Storage* _storage;
};

#endif // _SELFPLAY_H_