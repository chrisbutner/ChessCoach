#include "SelfPlay.h"

#include <limits>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <numeric>

#include <Stockfish/thread.h>
#include <Stockfish/uci.h>

#include "Config.h"

thread_local PoolAllocator<Node, Node::BlockSizeBytes> Node::Allocator;

std::atomic_uint SelfPlayGame::ThreadSeed;
thread_local std::default_random_engine SelfPlayGame::Random(
    std::random_device()() +
    static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) +
    ++ThreadSeed);

Node::Node(float setPrior)
    : mostVisitedChild(MOVE_NONE, nullptr)
    , originalPrior(setPrior)
    , prior(setPrior)
    , visitCount(0)
    , valueSum(0.f)
    , terminalValue(CHESSCOACH_VALUE_UNINITIALIZED)
{
    assert(!std::isnan(setPrior));
}

void* Node::operator new(size_t count)
{
    return Allocator.Allocate();
}

void Node::operator delete(void* ptr) noexcept
{
    Allocator.Free(ptr);
}

bool Node::IsExpanded() const
{
    return !children.empty();
}

float Node::Value() const
{
    return (visitCount > 0) ? (valueSum / visitCount) : CHESSCOACH_VALUE_LOSS;
}

// TODO: Write a custom allocator for nodes (work out very maximum, then do some kind of ring/tree - important thing is all same size, capped number)
// TODO: Also input/output planes? e.g. for StoredGames vector storage

// Fast default-constructor with no resource ownership, used to size out vectors.
SelfPlayGame::SelfPlayGame()
    : _root(nullptr)
    , _tryHard(false)
    , _image(nullptr)
    , _value(nullptr)
    , _policy(nullptr)
    , _searchRootPly(0)
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
    : Game()
    , _root(new Node(0.f))
    , _tryHard(false)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _searchRootPly(0)
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(const std::string& fen, const std::vector<Move>& moves, bool tryHard,
    INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
    : Game(fen, moves)
    , _root(new Node(0.f))
    , _tryHard(tryHard)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _searchRootPly(0)
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(const SelfPlayGame& other)
    : Game(other)
    , _root(other._root)
    , _tryHard(other._tryHard)
    , _image(other._image)
    , _value(other._value)
    , _policy(other._policy)
    , _searchRootPly(other.Ply())
    , _result(other._result)
{
    assert(&other != this);
}

SelfPlayGame& SelfPlayGame::operator=(const SelfPlayGame& other)
{
    assert(&other != this);

    Game::operator=(other);

    _root = other._root;
    _tryHard = other._tryHard;
    _image = other._image;
    _value = other._value;
    _policy = other._policy;
    _searchRootPly = other.Ply();
    _result = other._result;

    return *this;
}

SelfPlayGame::SelfPlayGame(SelfPlayGame&& other) noexcept
    : Game(other)
    , _root(other._root)
    , _tryHard(other._tryHard)
    , _image(other._image)
    , _value(other._value)
    , _policy(other._policy)
    , _searchRootPly(other._searchRootPly)
    , _childVisits(std::move(other._childVisits))
    , _history(std::move(other._history))
    , _result(other._result)
{
    assert(&other != this);
}

SelfPlayGame& SelfPlayGame::operator=(SelfPlayGame&& other) noexcept
{
    assert(&other != this);

    Game::operator=(static_cast<Game&&>(other));

    _root = other._root;
    _tryHard = other._tryHard;
    _image = other._image;
    _value = other._value;
    _policy = other._policy;
    _searchRootPly = other._searchRootPly;
    _childVisits = std::move(other._childVisits);
    _history = std::move(other._history);
    _result = other._result;

    return *this;
}

SelfPlayGame::~SelfPlayGame()
{
}

Node* SelfPlayGame::Root() const
{
    return _root;
}

bool SelfPlayGame::IsTerminal() const
{
    return (_root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED) || (Ply() >= Config::MaxMoves);
}

float SelfPlayGame::TerminalValue() const
{
    // Coalesce a draw for the (Ply >= MaxMoves) and other undetermined/unfinished cases.
    return (_root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED) ? _root->terminalValue : CHESSCOACH_VALUE_DRAW;
}

float SelfPlayGame::Result() const
{
    // Require that the caller has called Complete() before calling Result().
    assert(_result != CHESSCOACH_VALUE_UNINITIALIZED);
    return _result;
}

bool SelfPlayGame::TryHard() const
{
    return _tryHard;
}

void SelfPlayGame::ApplyMoveWithRoot(Move move, Node* newRoot)
{
    ApplyMove(move);
    _root = newRoot;

    // Don't adjust visit counts here because this is a common path; e.g. for scratch games also.
}

void SelfPlayGame::ApplyMoveWithRootAndHistory(Move move, Node* newRoot)
{
    ApplyMoveWithRoot(move, newRoot);
    _history.push_back(move);

    // Adjust the visit count for the new root so that it matches the sum of child visits from now on.
    // If the new root is a terminal node, reset to zero.
    // Otherwise, decrement because the node was visited exactly once as a leaf before being expanded.
    if (_root->children.empty())
    {
        _root->visitCount = 0;
    }
    else
    {
        _root->visitCount--;
    }
    assert(_root->visitCount == std::transform_reduce(_root->children.begin(), _root->children.end(),
        0, std::plus<>(), [](auto pair) { return pair.second->visitCount; }));
}

float SelfPlayGame::ExpandAndEvaluate(SelfPlayState& state, PredictionCacheEntry*& cacheStore)
{
    Node* root = _root;
    assert(!root->IsExpanded());

    // A known-terminal leaf will remain a leaf, so be prepared to
    // quickly return its terminal value on repeated visits.
    if (root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED)
    {
        state = SelfPlayState::Working;
        return root->terminalValue;
    }

    if (state == SelfPlayState::Working)
    {
        // Try get a cached prediction.
        cacheStore = nullptr;
        float cachedValue;
        int cachedMoveCount;
        _imageKey = GenerateImageKey();
        bool hitCached = false;
        if (Ply() <= Config::MaxPredictionCachePly)
        {
            hitCached = PredictionCache::Instance.TryGetPrediction(_imageKey, &cacheStore, &cachedValue,
                &cachedMoveCount, _cachedMoves.data(), _cachedPriors.data());
        }
        if (hitCached)
        {
            // Expand child nodes with the cached priors.
            int i = 0;
            for (int i = 0; i < cachedMoveCount; i++)
            {
                root->children[_cachedMoves[i]] = new Node(_cachedPriors[i]);
            }

            return cachedValue;
        }

        // Generate legal moves.
        _expandAndEvaluate_endMoves = generate<LEGAL>(_position, _expandAndEvaluate_moves);

        // Check for checkmate and stalemate.
        if (_expandAndEvaluate_moves == _expandAndEvaluate_endMoves)
        {
            root->terminalValue = (_position.checkers() ? CHESSCOACH_VALUE_LOSS : CHESSCOACH_VALUE_DRAW);
            return root->terminalValue;
        }

        // Check for draw by 50-move or 3-repetition.
        //
        // Stockfish checks for (a) two-fold repetition strictly after the search root
        // (e.g. search-root, rep-0, rep-1) or (b) three-fold repetition anywhere
        // (e.g. rep-0, rep-1, search-root, rep-2) in order to terminate and prune efficiently.
        //
        // We can use the same logic safely because we're path-dependent: no post-search valuations
        // are hashed purely by position (only network-dependent predictions, potentially),
        // and nodes with identical positions reached differently are distinct in the tree.
        //
        // This saves time in the 800-simulation budget for more useful exploration.
        const int plyToSearchRoot = (Ply() - _searchRootPly);
        if (IsDrawByNoProgressOrRepetition(plyToSearchRoot))
        {
            root->terminalValue = CHESSCOACH_VALUE_DRAW;
            return root->terminalValue;
        }

        // Prepare for a prediction from the network.
        *_image = GenerateImage();
        state = SelfPlayState::WaitingForPrediction;
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Received a prediction from the network.

    // Mix in the Stockfish evaluation when available.
    //
    // The important thing here is that this helps guide the MCTS search and thus the policy training,
    // but doesn't train the value head: that is still based purely on game result, so the network isn't
    // trying to learn a linear human evaluation function.
    if (!TryHard() && StockfishCanEvaluate())
    {
        // TODO: Lerp based on training progress.
        const float stockfishProbability01 = StockfishEvaluation();
        const float stockfishiness = 0.5f;
        *_value = (*_value * (1.f - stockfishiness)) + (stockfishProbability01 * stockfishiness);
    }

    // Index legal moves into the policy output planes to get logits,
    // then calculate softmax over them to get normalized probabilities for priors.
    int moveCount = 0;
    for (ExtMove* cur = _expandAndEvaluate_moves; cur != _expandAndEvaluate_endMoves; cur++)
    {
        _cachedMoves[moveCount] = cur->move;
        _cachedPriors[moveCount] = PolicyValue(*_policy, cur->move); // Logits
        moveCount++;
    }
    Softmax(moveCount, _cachedPriors.data()); // Logits -> priors

    // Store in the cache if appropriate. This may limit moveCount to the branch limit for caching.
    // In that case, better to also apply that limit now for consistency.
    if (cacheStore)
    {
        if (moveCount > Config::MaxBranchMoves)
        {
            LimitBranchingToBest(moveCount, _cachedMoves.data(), _cachedPriors.data());
            moveCount = Config::MaxBranchMoves;
        }
        cacheStore->Set(_imageKey, *_value, moveCount, _cachedMoves.data(), _cachedPriors.data());
    }

    // Expand child nodes with the calculated priors.
    for (int i = 0; i < moveCount; i++)
    {
        root->children[_cachedMoves[i]] = new Node(_cachedPriors[i]);
    }

    state = SelfPlayState::Working;
    return *_value;
}

void SelfPlayGame::LimitBranchingToBest(int moveCount, Move* moves, float* priors)
{
    assert(moveCount > Config::MaxBranchMoves);

    for (int i = 0; i < Config::MaxBranchMoves; i++)
    {
        int max = i;
        for (int j = i + 1; j < moveCount; j++)
        {
            if (priors[j] > priors[max]) max = j;
        }
        if (max != i)
        {
            std::swap(moves[i], moves[max]);
            std::swap(priors[i], priors[max]);
        }
    }
}

// Avoid Position::is_draw because it regenerates legal moves.
// If we've already just checked for checkmate and stalemate then this works fine.
bool SelfPlayGame::IsDrawByNoProgressOrRepetition(int plyToSearchRoot)
{
    const StateInfo& stateInfo = _positionStates->back();

    return 
        // Omit "and not checkmate" from Position::is_draw.
        (stateInfo.rule50 > 99) ||
        // Return a draw score if a position repeats once earlier but strictly
        // after the root, or repeats twice before or at the root.
        (stateInfo.repetition && (stateInfo.repetition < plyToSearchRoot));
}

void SelfPlayGame::Softmax(int moveCount, float* distribution) const
{
    const float max = *std::max_element(distribution, distribution + moveCount);

    float expSum = 0.f;
    for (int i = 0; i < moveCount; i++)
    {
        expSum += std::expf(distribution[i] - max);
    }

    const float logSumExp = std::logf(expSum) + max;
    for (int i = 0; i < moveCount; i++)
    {
        distribution[i] = std::expf(distribution[i] - logSumExp);
    }
}

std::pair<Move, Node*> SelfPlayGame::SelectMove() const
{
    if (!TryHard() && (Ply() < Config::NumSampingMoves))
    {
        // Use temperature=1; i.e., no need to exponentiate, just use visit counts as the distribution.
        const int sumChildVisits = _root->visitCount;
        int sample = std::uniform_int_distribution<int>(0, sumChildVisits - 1)(Random);
        for (const auto& pair : _root->children)
        {
            const int visitCount = pair.second->visitCount;
            if (sample < visitCount)
            {
                return pair;
            }
            sample -= visitCount;
        }
        assert(false);
        return std::pair(MOVE_NONE, nullptr);
    }
    else
    {
        // Use temperature=inf; i.e., just select the most visited.
        assert(_root->mostVisitedChild.second);
        return _root->mostVisitedChild;
    }
}

void SelfPlayGame::StoreSearchStatistics()
{
    std::map<Move, float> visits;
    const int sumChildVisits = _root->visitCount;
    for (const auto& pair : _root->children)
    {
        visits[pair.first] = static_cast<float>(pair.second->visitCount) / sumChildVisits;
    }
    _childVisits.emplace_back(std::move(visits));
}

void SelfPlayGame::Complete()
{
    // Save state that depends on nodes.
    _result = FlipValue(ToPlay(), TerminalValue());

    // Clear and detach from all nodes.
    PruneAll();
}

SavedGame SelfPlayGame::Save() const
{
    return SavedGame(Result(), _history, _childVisits);
}

void SelfPlayGame::PruneExcept(Node* root, Node* except)
{
    if (!root)
    {
        return;
    }

    // Rely on caller to already have updated the _root to the preserved subtree.
    assert(_root != root);
    assert(_root == except);

    for (auto& pair : root->children)
    {
        if (pair.second != except)
        {
            PruneAllInternal(pair.second);
        }
    }
    delete root;
}

void SelfPlayGame::PruneAll()
{
    if (!_root)
    {
        return;
    }

    PruneAllInternal(_root);

    // All nodes in the related tree are gone, so don't leave _root dangling.
    _root = nullptr;
}

void SelfPlayGame::PruneAllInternal(Node* root)
{
    for (auto& pair : root->children)
    {
        PruneAllInternal(pair.second);
    }
    delete root;
}

SelfPlayWorker::SelfPlayWorker()
    : _storage(nullptr)
    , _states(Config::PredictionBatchSize)
    , _images(Config::PredictionBatchSize)
    , _values(Config::PredictionBatchSize)
    , _policies(Config::PredictionBatchSize)
    , _games(Config::PredictionBatchSize)
    , _scratchGames(Config::PredictionBatchSize)
    , _gameStarts(Config::PredictionBatchSize)
    , _mctsSimulations(Config::PredictionBatchSize, 0)
    , _searchPaths(Config::PredictionBatchSize)
    , _cacheStores(Config::PredictionBatchSize)
    , _searchConfig{}
    , _searchState{}
{
}

void SelfPlayWorker::ResetGames()
{
    for (int i = 0; i < Config::PredictionBatchSize; i++)
    {
        SetUpGame(i);
    }
}

void SelfPlayWorker::PlayGames(WorkCoordinator& workCoordinator, Storage* storage, INetwork* network)
{
    // Initialize on the worker thread.
    Initialize(storage);

    while (true)
    {
        // Wait until games are required.
        workCoordinator.WaitForWorkItems();

        // Clear away old games in progress to ensure that new ones use the new network.
        ResetGames();

        // Play games until required.
        while (!workCoordinator.AllWorkItemsCompleted())
        {
            // CPU work
            for (int i = 0; i < _games.size(); i++)
            {
                Play(i);

                // In degenerate conditions whole games can finish in CPU via the prediction cache, so loop.
                while ((_states[i] == SelfPlayState::Finished) && !workCoordinator.AllWorkItemsCompleted())
                {
                    SaveToStorageAndLog(i);

                    workCoordinator.OnWorkItemCompleted();

                    SetUpGame(i);
                    Play(i);
                }
            }

            // GPU work
            network->PredictBatch(Config::PredictionBatchSize, _images.data(), _values.data(), _policies.data());
        }
    }
}

void SelfPlayWorker::Initialize(Storage* storage)
{
    _storage = storage;
}

void SelfPlayWorker::SetUpGame(int index)
{
    _states[index] = SelfPlayState::Working;
    _games[index] = SelfPlayGame(&_images[index], &_values[index], &_policies[index]);
    _gameStarts[index] = std::chrono::high_resolution_clock::now();
}

void SelfPlayWorker::SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard)
{
    _states[index] = SelfPlayState::Working;
    _games[index] = SelfPlayGame(fen, moves, tryHard, &_images[index], &_values[index], &_policies[index]);
    _gameStarts[index] = std::chrono::high_resolution_clock::now();
}

void SelfPlayWorker::DebugGame(INetwork* network, int index, const SavedGame& saved, int startingPly)
{
    SetUpGame(index);

    SelfPlayGame& game = _games[index];
    for (int i = 0; i < startingPly; i++)
    {
        game.ApplyMove(Move(saved.moves[i]));
    }

    while (true)
    {
        Play(index);
        assert(_states[index] == SelfPlayState::WaitingForPrediction);
        network->PredictBatch(index + 1, _images.data(), _values.data(), _policies.data());
    }
}

void SelfPlayWorker::TrainNetwork(INetwork* network, GameType gameType, int stepCount, int checkpoint) const
{
    // Train for "stepCount" steps.
    auto startTrain = std::chrono::high_resolution_clock::now();
    const int startStep = (checkpoint - stepCount + 1);
    for (int step = startStep; step <= checkpoint; step++)
    {
        TrainingBatch* batch = _storage->SampleBatch(gameType);
        network->TrainBatch(step, Config::BatchSize, batch->images.data(), batch->values.data(), batch->policies.data());

        // Test with one batch every TrainingStepsPerTest.
        if ((step % Config::TrainingStepsPerTest) == 0)
        {
            TrainingBatch* testBatch = _storage->SampleBatch(GameType_Test);
            network->TestBatch(step, Config::BatchSize, testBatch->images.data(), testBatch->values.data(), testBatch->policies.data());
        }
    }
    const float trainTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTrain).count();
    const float trainTimePerStep = (trainTime / stepCount);
    std::cout << "Trained steps " << startStep << "-" << checkpoint << ", total time " << trainTime << ", step time " << trainTimePerStep << std::endl;

    // Save the network and reload it for predictions.
    network->SaveNetwork(checkpoint);
}

void SelfPlayWorker::Play(int index)
{
    SelfPlayState& state = _states[index];
    SelfPlayGame& game = _games[index];

    if (!game.Root()->IsExpanded())
    {
        game.ExpandAndEvaluate(state, _cacheStores[index]);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }
    }

    while (!game.IsTerminal())
    {
        Node* root = game.Root();
        std::pair<Move, Node*> selected = RunMcts(game, _scratchGames[index], _states[index], _mctsSimulations[index], _searchPaths[index], _cacheStores[index]);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }

        assert(selected.second != nullptr);
        game.StoreSearchStatistics();
        game.ApplyMoveWithRootAndHistory(selected.first, selected.second);
        game.PruneExcept(root, selected.second /* == game.Root() */);
        _searchState.principleVariationChanged = true; // First move in PV is now gone.
    }

    // Clean up resources in use and save the result.
    game.Complete();

    state = SelfPlayState::Finished;
}

void SelfPlayWorker::SaveToStorageAndLog(int index)
{
    const SelfPlayGame& game = _games[index];

    const int ply = game.Ply();
    const float result = game.Result();
    const int gameNumber = _storage->AddGame(GameType_Train, game.Save());

    const float gameTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - _gameStarts[index]).count();
    const float mctsTime = (gameTime / ply);
    std::cout << "Game " << gameNumber << ", ply " << ply << ", time " << gameTime << ", mcts time " << mctsTime << ", result " << result << std::endl;
    //PredictionCache::Instance.PrintDebugInfo();
}

std::pair<Move, Node*> SelfPlayWorker::RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
    std::vector<std::pair<Move, Node*>>& searchPath, PredictionCacheEntry*& cacheStore)
{
    const int numSimulations = (game.TryHard() ? std::numeric_limits<int>::max() : Config::NumSimulations);
    for (; mctsSimulation < numSimulations; mctsSimulation++)
    {
        if (state == SelfPlayState::Working)
        {
            if (mctsSimulation == 0)
            {
#if !DEBUG_MCTS
                if (!game.TryHard())
                {
                    AddExplorationNoise(game);
                }
#endif

#if DEBUG_MCTS
                std::cout << "(Ready for ply " << game.Ply() << "...)" << std::endl;
                std::string _;
                std::getline(std::cin, _);
#endif
            }

            scratchGame = game;
            searchPath.clear();
            searchPath.push_back(std::pair(MOVE_NONE, scratchGame.Root()));

            while (scratchGame.Root()->IsExpanded())
            {
                std::pair<Move, Node*> selected = SelectChild(scratchGame.Root());
                scratchGame.ApplyMoveWithRoot(selected.first, selected.second);
                searchPath.push_back(selected /* == scratchGame.Root() */);
#if DEBUG_MCTS
                std::cout << Game::SquareName[from_sq(selected.first)] << Game::SquareName[to_sq(selected.first)] << "(" << selected.second->visitCount << "), ";
#endif
            }
        }

        float value = scratchGame.ExpandAndEvaluate(state, cacheStore);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return std::pair(MOVE_NONE, nullptr);
        }

        // The value we get is from the final node of the scratch game (could be WHITE or BLACK)
        // and we start applying it at the current position of the actual game (could again be WHITE or BLACK),
        // so flip it if they differ (the ^).
        //
        // Also though, always value a node from the parent's to-play perspective (the ~).
        //
        // E.g. imagine it's white to play (game.ToPlay()) and white makes the move a4,
        // which results in a new position with black to play (scratchGame.ToPlay()).
        // The network values this position as very bad for black (say 0.1). This means
        // it's very good for white (0.9), so white should continue visiting this child node.
        //
        // Or, imagine it's white to play and they have a mate-in-one. From black's perspective,
        // in the resulting position, it's a loss (0.0) because they're in check and have no moves,
        // thus no child nodes. This is a win for white (1.0), so white should continue visiting this
        // child node.
        //
        // It's important to keep the following values in sign/direction parity, for a single child position
        // (all should tend to be high, or all should tend to be low):
        // - visits
        // - network policy prediction (prior)
        // - network value prediction (valueSum / visitCount, back-propagated)
        // - terminal valuation (valueSum / visitCount, back-propagated)
        assert(!std::isnan(value));
        value = SelfPlayGame::FlipValue(Color(game.ToPlay() ^ ~scratchGame.ToPlay()), value);
        Backpropagate(searchPath, value);
        _searchState.nodeCount++;

#if DEBUG_MCTS
        std::cout << "prior " << scratchGame.Root()->originalPrior << ", noisy prior " << scratchGame.Root()->prior << ", prediction " << value << std::endl;
#endif
    }

    mctsSimulation = 0;
    return game.SelectMove();
}

void SelfPlayWorker::AddExplorationNoise(SelfPlayGame& game) const
{
    std::gamma_distribution<float> noise(Config::RootDirichletAlpha, 1.f);
    for (const auto& pair : game.Root()->children)
    {
        Node* child = pair.second;
        const float childNoise = noise(SelfPlayGame::Random);
        child->prior = child->originalPrior * (1 - Config::RootExplorationFraction) + childNoise * Config::RootExplorationFraction;
    }
}

std::pair<Move, Node*> SelfPlayWorker::SelectChild(const Node* parent) const
{
    float maxUcbScore = -std::numeric_limits<float>::infinity();
    std::pair<Move, Node*> max;
    for (const auto& pair : parent->children)
    {
        const float ucbScore = CalculateUcbScore(parent, pair.second);
        if (ucbScore > maxUcbScore)
        {
            maxUcbScore = ucbScore;
            max = pair;
        }
    }
    return max;
}

// TODO: Profile, see if significant, whether vectorizing is viable/worth it
float SelfPlayWorker::CalculateUcbScore(const Node* parent, const Node* child) const
{
    const float pbC = (std::logf((parent->visitCount + Config::PbCBase + 1.f) / Config::PbCBase) + Config::PbCInit) *
        std::sqrtf(static_cast<float>(parent->visitCount)) / (child->visitCount + 1.f);
    const float priorScore = pbC * child->prior;
    return priorScore + child->Value();
}

void SelfPlayWorker::Backpropagate(const std::vector<std::pair<Move, Node*>>& searchPath, float value)
{
    // Each ply has a different player, so flip each time.
    for (auto [move, node] : searchPath)
    {
        node->visitCount++;
        node->valueSum += value;
        value = SelfPlayGame::FlipValue(value);
    }

    // Adjust most-visited pointers (principle variation).
    bool isPrincipleVariation = true;
    for (int i = 0; i < searchPath.size() - 1; i++)
    {
        if (!searchPath[i].second->mostVisitedChild.second ||
            (searchPath[i].second->mostVisitedChild.second->visitCount < searchPath[i + 1].second->visitCount))
        {
            searchPath[i].second->mostVisitedChild = searchPath[i + 1];
            _searchState.principleVariationChanged |= isPrincipleVariation;
        }
        else
        {
            isPrincipleVariation &= (searchPath[i].second->mostVisitedChild == searchPath[i + 1]);
        }
    }

}

void SelfPlayWorker::DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut)
{
    *gameOut = &_games[index];
    *stateOut = &_states[index];
    *valuesOut = &_values[index];
    *policiesOut = &_policies[index];
}

SearchState& SelfPlayWorker::DebugSearchState()
{
    return _searchState;
}

void SelfPlayWorker::Search(std::function<INetwork*()> networkFactory)
{
    // Create the network on the worker thread (slow).
    std::unique_ptr<INetwork> network(networkFactory());

    // Warm up the GIL and predictions.
    network->PredictBatch(1, _images.data(), _values.data(), _policies.data());

    while (!_searchConfig.quit)
    {
        {
            std::unique_lock lock(_searchConfig.mutexUci);

            // Let UCI know we're ready.
            if (!_searchConfig.ready)
            {
                _searchConfig.ready = true;
                _searchConfig.signalReady.notify_all();
            }

            // Wait until told to search.
            while (!_searchConfig.quit && !_searchConfig.search)
            {
                _searchConfig.signalUci.wait(lock);
            }
        }

        UpdatePosition();
        UpdateSearch();
        if (_searchState.searching)
        {
            while (!_searchConfig.quit && !_searchConfig.positionUpdated && _searchState.searching)
            {
                SearchPlay(0);
                network->PredictBatch(1, _images.data(), _values.data(), _policies.data());

                CheckPrintInfo();

                // TODO: Only check every N times
                CheckTimeControl();

                UpdateSearch();
            }
            OnSearchFinished();
        }
    }

    // Clean up.
    _games[0].PruneAll();
}

void SelfPlayWorker::UpdatePosition()
{
    assert(!_searchState.searching);

    if (_searchConfig.positionUpdated)
    {
        std::lock_guard lock(_searchConfig.mutexUci);

        // Lock around both (a) using the position info, and (b) clearing the flag.
        // If the GUI does two updates very quickly, either (i) we grabbed the second one's
        // position info and cleared, or (ii) the flag gets set again after we unlock. Either way
        // we're good.

        _games[0].PruneAll();
        SetUpGame(0, _searchConfig.positionFen, _searchConfig.positionMoves, true /* tryHard */);

        _searchConfig.positionUpdated = false;
    }
}

void SelfPlayWorker::UpdateSearch()
{
    if (_searchConfig.searchUpdated)
    {
        std::lock_guard lock(_searchConfig.mutexUci);

        // Lock around both (a) using the search/time control info, and (b) clearing the flag.
        // If the GUI does two updates very quickly, either (i) we grabbed the second one's
        // search/time control info and cleared, or (ii) the flag gets set again after we unlock.
        // Either way we're good.

        _searchState.searching = _searchConfig.search;

        if (_searchState.searching)
        {
            _searchState.searchStart = std::chrono::high_resolution_clock::now();
            _searchState.timeControl = _searchConfig.searchTimeControl;
            _searchState.nodeCount = 0;
            _searchState.principleVariationChanged = 0;
        }

        // Set the "search" instruction to false now so that when this search finishes
        // the worker can go back to sleep, unless instructed to search again.
        // A stop command will still cause the "searchUpdated" flag to call in here and
        // set the "searching" state to false.
        _searchConfig.search = false;

        _searchConfig.searchUpdated = false;
    }
}

void SelfPlayWorker::OnSearchFinished()
{
    // We may have finished via position update or quit, so update our state.
    _searchState.searching = false;

    // Print the final PV info and bestmove.
    auto [move, node] = _games[0].SelectMove();
    PrintPrincipleVariation();
    std::cout << "bestmove " << UCI::move(move, false /* chess960 */) << std::endl;

    // Lock around (a) checking "searchUpdated" and (b) clearing "search". We want to clear
    // "search" in order to go back to sleep but only if it's still the existing search.
    {
        std::lock_guard lock(_searchConfig.mutexUci);

        if (!_searchConfig.searchUpdated)
        {
            _searchConfig.search = false;
        }
    }
}

void SelfPlayWorker::CheckPrintInfo()
{
    // Print principle variation when it changes.
    if (_searchState.principleVariationChanged)
    {
        PrintPrincipleVariation();
        _searchState.principleVariationChanged = false;
    }
}

void SelfPlayWorker::CheckTimeControl()
{
    // Always do at least 1-2 simulations so that a "best" move exists.
    if (!_games[0].Root()->mostVisitedChild.second)
    {
        return;
    }

    // Infinite think takes first priority.
    if (_searchState.timeControl.infinite)
    {
        return;
    }

    const std::chrono::duration sinceSearchStart = (std::chrono::high_resolution_clock::now() - _searchState.searchStart);
    const int64_t searchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(sinceSearchStart).count();

    // Specified think time takes second priority.
    if (_searchState.timeControl.moveTimeMs > 0)
    {
        if (searchTimeMs >= _searchState.timeControl.moveTimeMs)
        {
            _searchState.searching = false;
        }
        return;
    }

    // Game clock takes third priority. Use a simple strategy like AlphaZero for now.
    const Color toPlay = _games[0].ToPlay();
    const int64_t timeAllowed =
        (_searchState.timeControl.timeRemainingMs[toPlay] / Config::TimeControl_FractionOfRemaining)
        + _searchState.timeControl.incrementMs[toPlay]
        - Config::TimeControl_SafetyBufferMs;
    if (timeAllowed > 0)
    {
        if (searchTimeMs >= timeAllowed)
        {
            _searchState.searching = false;
        }
        return;
    }

    // No time allowed at all: defy the system and just make a quick training-style move.
    if (_mctsSimulations[0] >= Config::NumSimulations)
    {
        _searchState.searching = false;
    }
}

void SelfPlayWorker::PrintPrincipleVariation()
{
    Node* node = _games[0].Root();
    std::vector<Move> principleVariation;

    assert(node->mostVisitedChild.second);
    while (node->mostVisitedChild.second)
    {
        principleVariation.push_back(node->mostVisitedChild.first);
        node = node->mostVisitedChild.second;
    }

    const std::chrono::duration sinceSearchStart = (std::chrono::high_resolution_clock::now() - _searchState.searchStart);

    // Value is from the parent's perspective, so that's already correct for the root perspective,
    // but we need to flip from the root's color to white.
    const int depth = static_cast<int>(principleVariation.size());
    const float value = _games[0].Root()->mostVisitedChild.second->Value();
    const float whiteValue = Game::FlipValue(_games[0].ToPlay(), value);
    const int whiteScore = static_cast<int>(Game::ProbabilityToCentipawns(whiteValue));
    const int64_t searchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(sinceSearchStart).count();
    const int nodeCount = _searchState.nodeCount;
    const int nodesPerSecond = static_cast<int>(nodeCount / std::chrono::duration<float>(sinceSearchStart).count());

    std::cout << "info depth " << depth << " score cp " << whiteScore << " nodes " << nodeCount << " nps " << nodesPerSecond << " time " << searchTimeMs << " pv";
    for (Move move : principleVariation)
    {
        std::cout << " " << UCI::move(move, false /* chess960 */);
    }
    std::cout << std::endl;
}

void SelfPlayWorker::SignalDebug(bool debug)
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.debug = debug;
}

void SelfPlayWorker::SignalPosition(std::string&& fen, std::vector<Move>&& moves)
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.positionUpdated = true;
    _searchConfig.positionFen = std::move(fen);
    _searchConfig.positionMoves = std::move(moves);
}

void SelfPlayWorker::SignalSearchGo(const TimeControl& timeControl)
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.searchUpdated = true;
    _searchConfig.search = true;
    _searchConfig.searchTimeControl = timeControl;

    _searchConfig.signalUci.notify_all();
}

void SelfPlayWorker::SignalSearchStop()
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.searchUpdated = true;
    _searchConfig.search = false;
}

void SelfPlayWorker::SignalQuit()
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.quit = true;

    _searchConfig.signalUci.notify_all();
}

void SelfPlayWorker::WaitUntilReady()
{
    std::unique_lock lock(_searchConfig.mutexUci);

    while (!_searchConfig.ready)
    {
        _searchConfig.signalReady.wait(lock);
    }
}

void SelfPlayWorker::SearchPlay(int index)
{
    SelfPlayState& state = _states[index];
    SelfPlayGame& game = _games[index];

    if (!game.Root()->IsExpanded())
    {
        game.ExpandAndEvaluate(state, _cacheStores[index]);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }
    }

    RunMcts(game, _scratchGames[index], _states[index], _mctsSimulations[index], _searchPaths[index], _cacheStores[index]);
}