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
#include "PoolAllocator.h"
#include "Epd.h"

class TerminalValue
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

template <typename T>
class SiblingIterator
{
public:

    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = int;
    using pointer = T*;
    using reference = T&;

public:

    SiblingIterator(T* current)
        : _current(current)
    {
    }

    SiblingIterator& operator++()
    {
        _current = _current->nextSibling;
        return *this;
    }

    SiblingIterator operator++(int)
    {
        SiblingIterator pre = *this;
        _current = _current->nextSibling;
        return pre;
    }

    T& operator*()
    {
        return *_current;
    }

    T* operator->()
    {
        return _current;
    }

    bool operator==(const SiblingIterator& other)
    {
        return (_current == other._current);
    }

    bool operator!=(const SiblingIterator& other)
    {
        return (_current != other._current);
    }

private:

    T* _current;
};

struct Node
{
public:

    static const size_t BlockSizeBytes = 64 * 1024 * 1024; // 64 MiB
    thread_local static PoolAllocator<Node, BlockSizeBytes> Allocator;

    using iterator = SiblingIterator<Node>;
    using const_iterator = SiblingIterator<const Node>;

public:

    Node(uint16_t setMove, float setPrior);
    Node(Move setMove, float setPrior);

    void* operator new(size_t byteCount);
    void operator delete(void* memory) noexcept;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    bool IsExpanded() const;
    float Value(const NetworkConfig* config) const;
    void AdjustVisitCount(int newVisitCount);

    Node* Child(Move match);
    int CountChildren() const;

public:

    Node* bestChild;
    Node* firstChild;
    Node* nextSibling;
    uint16_t move;
    bool expanding;
    uint8_t visitingCount;
    float prior;
    int visitCount;
    float valueAverage;
    float valueWeight;
    TerminalValue terminalValue;
    // 4 bytes implicit padding
};
static_assert(sizeof(TerminalValue) == 8);
static_assert(sizeof(Node) == 56); // 52=56

struct WeightedNode
{
    Node* node;
    float weight;
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
};

struct SearchConfig
{
    std::mutex mutexUci;
    std::condition_variable signalUci;
    std::condition_variable signalReady;

    std::atomic_bool quit;
    std::atomic_bool debug;
    bool ready;
    std::atomic_bool comment;
    bool gui;

    std::atomic_bool searchUpdated;
    std::atomic_bool search;
    TimeControl searchTimeControl;

    std::atomic_bool positionUpdated;
    std::string positionFen;
    std::vector<Move> positionMoves;
};

struct SearchState
{
    std::string positionFen;
    std::vector<Move> positionMoves;
    bool searching;
    bool gui;
    std::chrono::time_point<std::chrono::high_resolution_clock> searchStart;
    std::chrono::time_point<std::chrono::high_resolution_clock>lastPrincipleVariationPrint;
    TimeControl timeControl;
    int nodeCount;
    int previousNodeCount;
    int failedNodeCount;
    bool principleVariationChanged;
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

    SelfPlayGame SpawnShadow(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy);

    Node* Root() const;
    float Result() const;

    bool TryHard() const;
    void ApplyMoveWithRoot(Move move, Node* newRoot);
    void ApplyMoveWithRootAndHistory(Move move, Node* newRoot);
    float ExpandAndEvaluate(SelfPlayState& state, PredictionCacheChunk*& cacheStore);
    void Expand(int moveCount);
    bool IsDrawByTwofoldRepetition(int plyToSearchRoot);
    void Softmax(int moveCount, float* distribution) const;
    float CalculateMctsValue(const NetworkConfig* config) const;
    void StoreSearchStatistics(const NetworkConfig* config);
    void Complete();
    SavedGame Save() const;
    void PruneExcept(Node* root, Node*& except);
    void PruneAll();
    void UpdateSearchRootPly();

    Move ParseSan(const std::string& san);

private:

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

class SelfPlayWorker
{
private:

    static Throttle PredictionCacheResetThrottle;

public:

    SelfPlayWorker(const NetworkConfig& networkConfig, Storage* storage);

    SelfPlayWorker(const SelfPlayWorker& other) = delete;
    SelfPlayWorker& operator=(const SelfPlayWorker& other) = delete;
    SelfPlayWorker(SelfPlayWorker&& other) = delete;
    SelfPlayWorker& operator=(SelfPlayWorker&& other) = delete;

    const NetworkConfig& Config() const;
    void PlayGames(WorkCoordinator& workCoordinator, INetwork* network);
    void ClearGame(int index);
    void SetUpGame(int index);
    void SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard);
    void SetUpGameExisting(int index, const std::vector<Move>& moves, int applyNewMovesOffset);
    void TrainNetwork(INetwork* network, NetworkType networkType, std::vector<GameType>& gameTypes,
        std::vector<Window>& trainingWindows, int step, int checkpoint);
    void TrainNetworkWithCommentary(INetwork* network, int step, int checkpoint);
    void SaveNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    bool StrengthTestNetwork(INetwork* network, NetworkType networkType, int checkpoint);
    void Play(int index);
    bool IsTerminal(const SelfPlayGame& game) const;
    void SaveToStorageAndLog(INetwork* network, int index);
    void PredictBatchUniform(int batchSize, INetwork::InputPlanes* images, float* values, INetwork::OutputPlanes* policies);
    Node* RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
        std::vector<WeightedNode>& searchPath, PredictionCacheChunk*& cacheStore);
    void AddExplorationNoise(SelfPlayGame& game) const;
    Node* SelectMove(const SelfPlayGame& game) const;
    template <bool ForcePlayouts>
    WeightedNode SelectChild(Node* node) const;
    template <bool ForcePlayouts>
    std::pair<float, float> CalculatePuctScore(const Node* parent, const Node* child) const;
    float CalculatePuctScoreFixedValue(const Node* parent, const Node* child, float fixedValue) const;
    void PrunePolicyTarget(Node* root) const;
    void Backpropagate(std::vector<WeightedNode>& searchPath, float value);
    void BackpropagateMate(const std::vector<WeightedNode>& searchPath);
    void FixPrincipleVariation(const std::vector<WeightedNode>& searchPath, Node* node);
    void UpdatePrincipleVariation(const std::vector<WeightedNode>& searchPath);
    void ValidatePrincipleVariation(const Node* root);
    bool WorseThan(const Node* lhs, const Node* rhs) const;
    void DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut);
    SearchState& DebugSearchState();
    void DebugResetGame(int index);

    void Search(std::function<INetwork* ()> networkFactory);
    PredictionStatus WarmUpPredictions(INetwork* network, NetworkType networkType, int batchSize);
    void SignalDebug(bool debug);
    void SignalPosition(std::string&& fen, std::vector<Move>&& moves);
    void SignalSearchGo(const TimeControl& timeControl);
    void SignalSearchStop();
    void SignalQuit();
    void SignalComment();
    void SignalGui();
    void WaitUntilReady();

    void StrengthTest(INetwork* network, NetworkType networkType, int step);
    std::tuple<int, int, int, int> StrengthTestEpd(INetwork* network, NetworkType networkType, const std::filesystem::path& epdPath,
        int moveTimeMs, int nodes, int failureNodes, int positionLimit,
        std::function<void(const std::string&, const std::string&, const std::string&, int, int, int)> progress);

private:

    void UpdatePosition();
    void UpdateSearch();
    void OnSearchFinished();
    void CheckPrintInfo();
    void CheckUpdateGui(INetwork* network, bool forceUpdate);
    void CheckTimeControl();
    void PrintPrincipleVariation();
    void SearchInitialize(int mctsParallelism);
    void SearchPlay(int mctsParallelism);
    void CommentOnPosition(INetwork* network);

    std::tuple<Move, int, int> StrengthTestPosition(INetwork* network, NetworkType networkType, const StrengthTestSpec& spec, int moveTimeMs, int nodes, int failureNodes);
    std::pair<int, int> JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move, int lastBestNodes, int failureNodes);

private:

    const NetworkConfig* _networkConfig;
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

    SearchConfig _searchConfig;
    SearchState _searchState;
};

#endif // _SELFPLAY_H_