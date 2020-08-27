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

    static int Draw();

    // Mate in N fullmoves, not halfmoves/ply.
    static int MateIn(int n);

    // Opponent mate in N fullmoves, not halfmoves/ply.
    static int OpponentMateIn(int n);

    // Mate in N fullmoves, not halfmoves/ply.
    template <int N>
    static constexpr int MateIn()
    {
        return N;
    }

    // Opponent mate in N fullmoves, not halfmoves/ply.
    template <int N>
    static constexpr int OpponentMateIn()
    {
        return -N;
    }

public:

    TerminalValue();
    TerminalValue(const int value);
    
    TerminalValue& operator=(const int value);
    bool operator==(const int other) const;

    bool IsNonTerminal() const;

    bool IsImmediate() const;
    float ImmediateValue() const;

    bool IsMateInN() const;
    bool IsOpponentMateInN() const;

    int MateN() const;
    int OpponentMateN() const;
    int EitherMateN() const;

    float MateScore(float explorationRate) const;

private:

    std::optional<int> _value;
    std::function<float(float)> _mateScore;
};

class Node
{
public:

    static const size_t BlockSizeBytes = 64 * 1024 * 1024; // 64 MiB
    thread_local static PoolAllocator<Node, BlockSizeBytes> Allocator;

public:

    Node(float setPrior);

    void* operator new(size_t byteCount);
    void operator delete(void* memory) noexcept;

    bool IsExpanded() const;
    float Value() const;

public:

    std::map<Move, Node*> children;
    std::pair<Move, Node*> bestChild;
    float prior;
    int visitCount;
    int visitingCount;
    float valueSum;
    TerminalValue terminalValue;
    bool expanding;
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
    std::chrono::time_point<std::chrono::high_resolution_clock> searchStart;
    std::chrono::time_point<std::chrono::high_resolution_clock>lastPrincipleVariationPrint;
    TimeControl timeControl;
    int nodeCount;
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
    bool IsDrawByNoProgressOrThreefoldRepetition();
    bool IsDrawByTwofoldRepetition(int plyToSearchRoot);
    void Softmax(int moveCount, float* distribution) const;
    void StoreSearchStatistics();
    void Complete();
    SavedGame Save() const;
    void PruneExcept(Node* root, Node* except);
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
public:

    SelfPlayWorker(const NetworkConfig& networkConfig, Storage* storage);

    SelfPlayWorker(const SelfPlayWorker& other) = delete;
    SelfPlayWorker& operator=(const SelfPlayWorker& other) = delete;
    SelfPlayWorker(SelfPlayWorker&& other) = delete;
    SelfPlayWorker& operator=(SelfPlayWorker&& other) = delete;

    const NetworkConfig& Config() const;
    void ResetGames();
    void PlayGames(WorkCoordinator& workCoordinator, INetwork* network);
    void ClearGame(int index);
    void SetUpGame(int index);
    void SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard);
    void SetUpGameExisting(int index, const std::vector<Move>& moves, int applyNewMovesOffset);
    void TrainNetwork(INetwork* network, int stepCount, int checkpoint);
    void ValidateNetwork(INetwork* network, int step);
    void Play(int index);
    bool IsTerminal(const SelfPlayGame& game) const;
    void SaveToStorageAndLog(int index);
    std::pair<Move, Node*> RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
        std::vector<std::pair<Move, Node*>>& searchPath, PredictionCacheChunk*& cacheStore);
    void AddExplorationNoise(SelfPlayGame& game) const;
    std::pair<Move, Node*> SelectMove(const SelfPlayGame& game) const;
    std::pair<Move, Node*> SelectChild(const Node* node) const;
    float CalculateUcbScore(const Node* parent, const Node* child) const;
    void Backpropagate(const std::vector<std::pair<Move, Node*>>& searchPath, float value);
    void BackpropagateMate(const std::vector<std::pair<Move, Node*>>& searchPath);
    void FixPrincipleVariation(const std::vector<std::pair<Move, Node*>>& searchPath, Node* node);
    void UpdatePrincipleVariation(const std::vector<std::pair<Move, Node*>>& searchPath);
    void ValidatePrincipleVariation(const Node* root);
    bool WorseThan(const Node* lhs, const Node* rhs) const;
    void DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut);
    SearchState& DebugSearchState();

    void Search(std::function<INetwork* ()> networkFactory);
    void WarmUpPredictions(INetwork* network, int batchSize);
    void SignalDebug(bool debug);
    void SignalPosition(std::string&& fen, std::vector<Move>&& moves);
    void SignalSearchGo(const TimeControl& timeControl);
    void SignalSearchStop();
    void SignalQuit();
    void WaitUntilReady();

    void StrengthTest(INetwork* network, int step);
    std::tuple<int, int, int> StrengthTest(INetwork* network, const std::filesystem::path& epdPath, int moveTimeMs);

private:

    void UpdatePosition();
    void UpdateSearch();
    void OnSearchFinished();
    void CheckPrintInfo();
    void CheckTimeControl();
    void PrintPrincipleVariation();
    void SearchInitialize(int mctsParallelism);
    void SearchPlay(int mctsParallelism);

    int StrengthTestPosition(INetwork* network, const StrengthTestSpec& spec, int moveTimeMs);
    int JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move);

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
    std::vector<std::vector<std::pair<Move, Node*>>> _searchPaths;
    std::vector<PredictionCacheChunk*> _cacheStores;

    SearchConfig _searchConfig;
    SearchState _searchState;
};

#endif // _SELFPLAY_H_