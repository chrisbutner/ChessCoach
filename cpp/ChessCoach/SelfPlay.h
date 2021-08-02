// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#ifndef _SELFPLAY_H_
#define _SELFPLAY_H_

#include <map>
#include <vector>
#include <atomic>
#include <functional>
#include <optional>

#include <Stockfish/position.h>
#include <Stockfish/movegen.h>

#include "Game.h"
#include "Network.h"
#include "Storage.h"
#include "SavedGame.h"
#include "Threading.h"
#include "PredictionCache.h"
#include "Epd.h"

class TerminalValue
{
public:

    static int8_t Draw();

    // Mate in N fullmoves, not halfmoves/ply.
    static int8_t MateIn(int8_t n);

    // Opponent mate in N fullmoves, not halfmoves/ply.
    static int8_t OpponentMateIn(int8_t n);

    // Mate in N fullmoves, not halfmoves/ply.
    template <int8_t N>
    static constexpr int8_t MateIn()
    {
        return N;
    }

    // Opponent mate in N fullmoves, not halfmoves/ply.
    template <int8_t N>
    static constexpr int8_t OpponentMateIn()
    {
        return -N;
    }

public:

    TerminalValue();
    TerminalValue(const int8_t value);

    TerminalValue& operator=(const int8_t value);
    bool operator==(const int8_t other) const;

    bool IsNonTerminal() const;

    bool IsImmediate() const;
    float ImmediateValue() const;

    bool IsMateInN() const;
    bool IsOpponentMateInN() const;

    int8_t MateN() const;
    int8_t OpponentMateN() const;
    int8_t EitherMateN() const;

    float MateScore(float explorationRate) const;

private:

    std::optional<int8_t> _value;
};

enum class Expansion : uint8_t
{
    None = 0,
    Expanding,
    Expanded,
};

struct alignas(64) Node
{
public:

    using iterator = Node*;
    using const_iterator = const Node*;

public:

    Node();
    Node(const Node& other);
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    bool IsExpanded() const;
    float Value() const;
    float ValueWithVirtualLoss() const;
    int SampleValue(float movingAverageBuild, float movingAverageCap, float value);
    float TablebaseBoundedValue(float value) const;
    void SetTerminalValue(TerminalValue value);
    void SetTablebaseScoreBound(float score, Bound bound);

    Node* Child(Move match);

public:

    std::atomic<Node*> bestChild;
    Node* children;

    int32_t childCount;
    float prior;
    uint16_t move;
    std::atomic_uint16_t visitingCount;
    std::atomic_int visitCount;

    std::atomic<float> valueAverage;
    std::atomic_int valueWeight;
    std::atomic_int upWeight;
    std::atomic<TerminalValue> terminalValue;
    std::atomic<Expansion> expansion;
    uint8_t padding1[1];

    std::atomic_int tablebaseRank;
    std::atomic<float> tablebaseScore;
    std::atomic<Bound> tablebaseBound;
    uint8_t padding2[4];
};
static_assert(sizeof(TerminalValue) == 2);
static_assert(sizeof(Expansion) == 1);
static_assert(sizeof(Node) == 64);
static_assert(alignof(Node) == 64);

struct WeightedNode
{
    Node* node;
    int weight;
};

struct ScoredNode
{
    Node* node;
    float score;
    float virtualExploration;

    ScoredNode(Node* setNode, float setScore, float setVirtualExploration)
        : node(setNode)
        , score(setScore)
        , virtualExploration(setVirtualExploration)
    {
    }

    bool operator<(const ScoredNode& other) const
    {
        return score > other.score;
    }
};

class SelfPlayWorker;
struct SearchState;

class PuctContext
{
public:

    PuctContext(const SearchState* searchState, Node* parent);
    WeightedNode SelectChild() const;
    float CalculatePuctScoreAdHoc(const Node* child) const;

private:

    thread_local static std::vector<ScoredNode> ScoredNodes;

private:

    float CalculateAzPuctScore(const Node* child, float childVirtualExploration) const;
    float CalculateSblePuctScore(float azPuctScore, float childVirtualExploration) const;
    float VirtualExploration(const Node* node) const;

private:

    Node* _parent;
    float _parentVirtualExploration;
    float _explorationNumerator;
    int _eliminationTopCount;
    float _linearExplorationRate;
    float _linearExplorationBase;
};

enum class SelfPlayState
{
    Working,
    WaitingForPrediction,
    Finished,
};

struct TimeControl
{
    bool pondering;
    bool infinite;
    int nodes;
    int mate;
    int64_t moveTimeMs;

    int64_t timeRemainingMs[COLOR_NB];
    int64_t incrementMs[COLOR_NB];
    int movesToGo;

    float eliminationFraction;
    int eliminationRootVisitCount;
};

class SelfPlayGame : public Game
{
public:

    SelfPlayGame();
    SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy, int* tablebaseCardinality);
    SelfPlayGame(const std::string& fen, const std::vector<Move>& moves, bool tryHard, INetwork::InputPlanes* image,
        float* value, INetwork::OutputPlanes* policy, int* tablebaseCardinality);

    SelfPlayGame(const SelfPlayGame& other);
    SelfPlayGame& operator=(const SelfPlayGame& other);
    SelfPlayGame(SelfPlayGame&& other) noexcept;
    SelfPlayGame& operator=(SelfPlayGame&& other) noexcept;
    ~SelfPlayGame();

    SelfPlayGame SpawnShadow(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy) const;

    Node* Root() const;
    float Result() const;

    bool TryHard() const;
    void ApplyMoveWithRoot(Move move, Node* newRoot);
    void ApplyMoveWithRootAndExpansion(Move move, Node* newRoot, SelfPlayWorker& selfPlayWorker);
    float ExpandAndEvaluate(SelfPlayState& state, PredictionCacheChunk*& cacheStore, SearchState* searchState,
        bool isSearchRoot, bool generateUniformPredictions);

    void PruneExcept(Node* root, Node*& except);
    void PruneAll();
    Move ParseSan(const std::string& san);
    void AddExplorationNoise();
    void UpdateSearchRootPly();
    bool ShouldProbeTablebases();
    int& TablebaseCardinality();

    void StoreSearchStatistics();
    void Complete();
    SavedGame Save() const;

    void DebugExpandCanonicalOrdering();

private:

    bool TakeExpansionOwnership(Node* node);
    void PruneAllInternal(Node* root);
    float FinishExpanding(SelfPlayState& state, PredictionCacheChunk*& cacheStore, SearchState* searchState, bool isSearchRoot, int moveCount, float value);
    void Expand(int moveCount, float firstPlayUrgency);

    bool IsDrawByTwofoldRepetition(int plyToSearchRoot);
    void Softmax(int moveCount, float* distribution) const;
    float CalculateMctsValue() const;

private:

    // Used for both real and scratch games.
    Node* _root;
    bool _tryHard;
    INetwork::InputPlanes* _image;
    float* _value;
    INetwork::OutputPlanes* _policy;
    int* _tablebaseCardinality;
    int _searchRootPly;

    // Stored history and statistics.
    // Only used for real games, so no need to copy, but may make sense for primitives.
    std::vector<float> _mctsValues;
    std::vector<std::map<Move, float>> _childVisits;
    float _result;

    // Coroutine state.
    // Only used for real games, so no need to copy.
    ExtMove _expandAndEvaluate_moves[MAX_MOVES];
    ExtMove* _expandAndEvaluate_endMoves;
    Key _imageKey;
    std::array<float, MAX_MOVES> _cachedPriors;
};

struct SearchState
{
    void Reset(const TimeControl& setTimeControl, std::chrono::time_point<std::chrono::high_resolution_clock> setSearchStart);

    // Controller + primary worker
    bool gui;
    std::string botGameId;
    std::string positionFen;
    std::vector<Move> positionMoves;
    std::vector<Move> searchMoves; // Not *necessarily* the primary worker, but only the first worker to expand the root.
    std::chrono::time_point<std::chrono::high_resolution_clock> searchStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastPrincipalVariationPrint;
    uint16_t lastBestMove;
    int lastBestNodes;
    TimeControl timeControl;
    int previousNodeCount;
    std::string guiLine;
    std::vector<Move> guiLineMoves;

    // All workers
    SelfPlayGame* position;
    std::atomic_bool debug;
    std::atomic_int nodeCount;
    std::atomic_int failedNodeCount;
    std::atomic_int tablebaseHitCount;
    std::atomic_bool principalVariationChanged;
};

class SelfPlayWorker
{
private:

    static Throttle PredictionCacheResetThrottle;

public:

    SelfPlayWorker(Storage* storage, SearchState* searchState, int gameCount);

    SelfPlayWorker(const SelfPlayWorker& other) = delete;
    SelfPlayWorker& operator=(const SelfPlayWorker& other) = delete;
    SelfPlayWorker(SelfPlayWorker&& other) = delete;
    SelfPlayWorker& operator=(SelfPlayWorker&& other) = delete;

    void Initialize();

    void LoopSelfPlay(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int threadIndex);
    void LoopSearch(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int threadIndex);
    void LoopStrengthTest(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int threadIndex);

    void SetUpGame(int index, const std::chrono::time_point<std::chrono::high_resolution_clock>& now);
    void SetUpGame(int index, const std::chrono::time_point<std::chrono::high_resolution_clock>& now, const std::string& fen, const std::vector<Move>& moves, bool tryHard);
    void SetUpGameExisting(int index, const std::chrono::time_point<std::chrono::high_resolution_clock>& now, const std::vector<Move>& moves, int applyNewMovesOffset);
    void TrainNetwork(INetwork* network, NetworkType networkType, int step, int checkpoint);
    void TrainNetworkWithCommentary(INetwork* network, int step, int checkpoint);
    void SaveNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    void SaveSwaNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    void StrengthTestNetwork(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int checkpoint);
    void BackpropagateMate(const std::vector<WeightedNode>& searchPath);
    bool WorseThan(const Node* lhs, const Node* rhs) const;
    void SearchUpdatePosition(const std::string& fen, const std::vector<Move>& moves, bool forceNewPosition);
    void CommentOnPosition(INetwork* network);
    void GuiShowLine(INetwork* network, const std::string& line);
    void Play(int index);
    Node* SelectMove(const SelfPlayGame& game, bool allowDiversity) const;
    void PrepareExpandedRoot(SelfPlayGame& game);
    std::tuple<int, int, int, int> StrengthTestEpd(WorkCoordinator* workCoordinator, const std::filesystem::path& epdPath,
        int moveTimeMs, int nodes, int failureNodes, int positionLimit,
        std::function<void(const std::string&, const std::string&, const std::string&, int, int, int)> progress);

    void DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut);
    void DebugResetGame(int index);

private:

    void Finalize();
    void FailNode(std::vector<WeightedNode>& searchPath);

    void FinalizeMcts();
    Move OnSearchFinished();
    void CheckPrincipalVariation();
    void CheckUpdateGui(INetwork* network, bool forceUpdate);
    void CheckTimeControl(WorkCoordinator* workCoordinator);
    void PrintPrincipalVariation(bool searchFinished);
    void SearchInitialize(const SelfPlayGame* position);
    bool SearchPlay(int threadIndex);

    std::tuple<Move, int, int> StrengthTestPosition(WorkCoordinator* workCoordinator, const StrengthTestSpec& spec, int moveTimeMs, int nodes, int failureNodes);
    std::pair<int, int> JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move, int lastBestNodes, int failureNodes);

    int ChooseSimulationLimit();
    void ClearGame(int index, const std::chrono::time_point<std::chrono::high_resolution_clock>& now);
    bool IsTerminal(const SelfPlayGame& game) const;
    void SaveToStorageAndLog(INetwork* network, int index);
    void PredictBatchUniform(int batchSize, INetwork::InputPlanes* images, float* values, INetwork::OutputPlanes* policies);
    bool RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation, int& mctsSimulationLimit,
        std::vector<WeightedNode>& searchPath, PredictionCacheChunk*& cacheStore, bool finishOnly);
    void Backpropagate(std::vector<WeightedNode>& searchPath, float value, float rootValue);
    void BackpropagateVisitsOnly(std::vector<WeightedNode>& searchPath, int index);
    void FixPrincipalVariation(const std::vector<WeightedNode>& searchPath, Node* node);
    void UpdatePrincipalVariation(const std::vector<WeightedNode>& searchPath);
    void ValidatePrincipalVariation(const Node* root);
    
    std::vector<Node*> CollectBestMoves(Node* parent, float valueDeltaThreshold) const;

    void UpdateGameForNewSearchRoot(SelfPlayGame& game);
    PredictionStatus WarmUpPredictions(INetwork* network, NetworkType networkType, int batchSize);

private:

    Storage* _storage;

    bool _generateUniformPredictions;
    std::vector<SelfPlayState> _states;
    std::vector<INetwork::InputPlanes> _images;
    std::vector<float> _values;
    std::vector<INetwork::OutputPlanes> _policies;
    std::vector<int> _tablebaseCardinalities;

    std::vector<SelfPlayGame> _games;
    std::vector<SelfPlayGame> _scratchGames;
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> _gameStarts;
    std::vector<int> _mctsSimulations;
    std::vector<int> _mctsSimulationLimits;
    std::vector<std::vector<WeightedNode>> _searchPaths;
    std::vector<PredictionCacheChunk*> _cacheStores;

    SearchState* _searchState;

    int _currentParallelism;
};

#endif // _SELFPLAY_H_