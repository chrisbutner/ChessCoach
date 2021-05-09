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

// Add "alignas(8)" to side-step ABI back-compatibility issue on Windows.
class alignas(8) TerminalValue
{
public:

    static TerminalValue NonTerminal();

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
    float _mateTerm;
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
    std::atomic<Expansion> expansion;
    uint8_t padding[11];
    std::atomic<TerminalValue> terminalValue;
};
static_assert(sizeof(TerminalValue) == 8);
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

class PuctContext
{
public:

    PuctContext(Node* parent);
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
    bool infinite;
    int nodes;
    int mate;
    int64_t moveTimeMs;

    int64_t timeRemainingMs[COLOR_NB];
    int64_t incrementMs[COLOR_NB];
    int movesToGo;
};

class SelfPlayGame : public Game
{
public:

    SelfPlayGame();
    SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy);
    SelfPlayGame(const std::string& fen, const std::vector<Move>& moves, bool tryHard, INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy);

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
    void ApplyMoveWithRootAndHistory(Move move, Node* newRoot);
    float ExpandAndEvaluate(SelfPlayState& state, PredictionCacheChunk*& cacheStore, float firstPlayUrgency);
    void Expand(int moveCount, float firstPlayUrgency);
    bool IsDrawByTwofoldRepetition(int plyToSearchRoot);
    void Softmax(int moveCount, float* distribution) const;
    float CalculateMctsValue() const;
    void StoreSearchStatistics();
    void Complete();
    SavedGame Save() const;
    void PruneExcept(Node* root, Node*& except);
    void PruneAll();
    void UpdateGameForNewSearchRoot();

    Move ParseSan(const std::string& san);

private:

    bool TakeExpansionOwnership(Node* node);
    void PruneAllInternal(Node* root);

private:

    // Used for both real and scratch games.
    Node* _root;
    bool _tryHard;
    INetwork::InputPlanes* _image;
    float* _value;
    INetwork::OutputPlanes* _policy;
    int _searchRootPly;

    // Stored history and statistics.
    // Only used for real games, so no need to copy, but may make sense for primitives.
    std::vector<float> _mctsValues;
    std::vector<std::map<Move, float>> _childVisits;
    std::vector<Move> _history;
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
    void Reset(const TimeControl& setTimeControl);

    // Controller + primary worker
    bool gui;
    std::string positionFen;
    std::vector<Move> positionMoves;
    std::chrono::time_point<std::chrono::high_resolution_clock> searchStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastPrincipleVariationPrint;
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
    std::atomic_bool principleVariationChanged;
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

    void LoopSelfPlay(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, bool primary);
    void LoopSearch(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, bool primary);
    void LoopStrengthTest(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, bool primary);

    void ClearGame(int index);
    void SetUpGame(int index);
    void SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard);
    void SetUpGameExisting(int index, const std::vector<Move>& moves, int applyNewMovesOffset);
    void TrainNetwork(INetwork* network, NetworkType networkType, int step, int checkpoint);
    void TrainNetworkWithCommentary(INetwork* network, int step, int checkpoint);
    void SaveNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    void SaveSwaNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    void StrengthTestNetwork(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int checkpoint);
    void Play(int index);
    bool IsTerminal(const SelfPlayGame& game) const;
    void SaveToStorageAndLog(INetwork* network, int index);
    void PredictBatchUniform(int batchSize, INetwork::InputPlanes* images, float* values, INetwork::OutputPlanes* policies);
    Node* RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
        std::vector<WeightedNode>& searchPath, PredictionCacheChunk*& cacheStore);
    void AddExplorationNoise(SelfPlayGame& game) const;
    Node* SelectMove(const SelfPlayGame& game, bool allowDiversity) const;
    void Backpropagate(std::vector<WeightedNode>& searchPath, float value, float rootValue);
    void BackpropagateMate(const std::vector<WeightedNode>& searchPath);
    void FixPrincipleVariation(const std::vector<WeightedNode>& searchPath, Node* node);
    void UpdatePrincipleVariation(const std::vector<WeightedNode>& searchPath);
    void ValidatePrincipleVariation(const Node* root);
    bool WorseThan(const Node* lhs, const Node* rhs) const;
    std::vector<Node*> CollectBestMoves(Node* parent, float valueDeltaThreshold) const;
    void DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut);
    void DebugResetGame(int index);

    void SearchUpdatePosition(const std::string& fen, const std::vector<Move>& moves, bool forceNewPosition);
    void CommentOnPosition(INetwork* network);
    PredictionStatus WarmUpPredictions(INetwork* network, NetworkType networkType, int batchSize);

    void GuiShowLine(INetwork* network, const std::string& line);

    std::tuple<int, int, int, int> StrengthTestEpd(WorkCoordinator* workCoordinator, const std::filesystem::path& epdPath,
        int moveTimeMs, int nodes, int failureNodes, int positionLimit,
        std::function<void(const std::string&, const std::string&, const std::string&, int, int, int)> progress);

private:

    void Finalize();
    void FailNode(std::vector<WeightedNode>& searchPath);

    void FinishMcts();
    void OnSearchFinished();
    void CheckPrincipleVariation();
    void CheckUpdateGui(INetwork* network, bool forceUpdate);
    void CheckTimeControl(WorkCoordinator* workCoordinator);
    void PrintPrincipleVariation(bool searchFinished);
    void SearchInitialize(const SelfPlayGame* position);
    void SearchPlay();

    std::tuple<Move, int, int> StrengthTestPosition(WorkCoordinator* workCoordinator, const StrengthTestSpec& spec, int moveTimeMs, int nodes, int failureNodes);
    std::pair<int, int> JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move, int lastBestNodes, int failureNodes);

private:

    Storage* _storage;

    std::vector<SelfPlayState> _states;
    std::vector<INetwork::InputPlanes> _images;
    std::vector<float> _values;
    std::vector<INetwork::OutputPlanes> _policies;

    std::vector<SelfPlayGame> _games;
    std::vector<SelfPlayGame> _scratchGames;
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> _gameStarts;
    std::vector<int> _mctsSimulations;
    std::vector<std::vector<WeightedNode>> _searchPaths;
    std::vector<PredictionCacheChunk*> _cacheStores;

    SearchState* _searchState;
};

#endif // _SELFPLAY_H_