#include "SelfPlay.h"

#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>

#include <Stockfish/thread.h>
#include <Stockfish/evaluate.h>

#include "Config.h"

thread_local PoolAllocator<Node> Node::Allocator;

std::atomic_uint SelfPlayGame::ThreadSeed;
thread_local std::default_random_engine SelfPlayGame::Random(
    std::random_device()() +
    static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) +
    ++ThreadSeed);

Node::Node(float setPrior)
    : originalPrior(setPrior)
    , prior(setPrior)
    , visitCount(0)
    , valueSum(0.f)
    , terminalValue(CHESSCOACH_VALUE_UNINITIALIZED)
    , _sumChildVisits(0)
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
    return (visitCount > 0) ? (valueSum / visitCount) : CHESSCOACH_VALUE_LOSE;
}

int Node::SumChildVisits() const
{
    if (_sumChildVisits == 0)
    {
        for (const auto& pair : children)
        {
            _sumChildVisits += pair.second->visitCount;
        }
        assert(_sumChildVisits > 0);
    }
    return _sumChildVisits;
}

// TODO: Write a custom allocator for nodes (work out very maximum, then do some kind of ring/tree - important thing is all same size, capped number)
// TODO: Also input/output planes? e.g. for StoredGames vector storage

// Fast default-constructor with no resource ownership, used to size out vectors.
SelfPlayGame::SelfPlayGame()
    : _root(nullptr)
    , _image(nullptr)
    , _value(nullptr)
    , _policy(nullptr)
    , _searchRootPly(0)
{
}

SelfPlayGame::SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
    : Game()
    , _root(new Node(0.f))
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _searchRootPly(0)
{
}

SelfPlayGame::SelfPlayGame(const SelfPlayGame& other)
    : Game(other)
    , _root(other._root)
    , _image(other._image)
    , _value(other._value)
    , _policy(other._policy)
    , _searchRootPly(other.Ply())
{
    assert(&other != this);
}

SelfPlayGame& SelfPlayGame::operator=(const SelfPlayGame& other)
{
    assert(&other != this);

    Game::operator=(other);

    _root = other._root;
    _image = other._image;
    _value = other._value;
    _policy = other._policy;
    _searchRootPly = other.Ply();

    return *this;
}

SelfPlayGame::SelfPlayGame(SelfPlayGame&& other) noexcept
    : Game(other)
    , _root(other._root)
    , _image(other._image)
    , _value(other._value)
    , _policy(other._policy)
    , _searchRootPly(other._searchRootPly)
    , _childVisits(std::move(other._childVisits))
    , _history(std::move(other._history))
{
    assert(&other != this);
}

SelfPlayGame& SelfPlayGame::operator=(SelfPlayGame&& other) noexcept
{
    assert(&other != this);

    Game::operator=(static_cast<Game&&>(other));

    _root = other._root;
    _image = other._image;
    _value = other._value;
    _policy = other._policy;
    _searchRootPly = other._searchRootPly;
    _childVisits = std::move(other._childVisits);
    _history = std::move(other._history);

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
    // Require that the caller has seen IsTerminal() as true before calling TerminalValue().
    // So, just coalesce a draw for the Ply >= MaxMoves case.
    return (_root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED) ? _root->terminalValue : CHESSCOACH_VALUE_DRAW;
}

float SelfPlayGame::Result() const
{
    // Require that the caller has seen IsTerminal() as true before calling Result(), and hasn't played/unplayed any moves.
    return FlipValue(ToPlay(), TerminalValue());
}

void SelfPlayGame::ApplyMoveWithRoot(Move move, Node* newRoot)
{
    ApplyMove(move);
    _root = newRoot;
}

void SelfPlayGame::ApplyMoveWithRootAndHistory(Move move, Node* newRoot)
{
    ApplyMoveWithRoot(move, newRoot);
    _history.push_back(move);
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
        bool hitCached = PredictionCache::Instance.TryGetPrediction(_imageKey, &cacheStore, &cachedValue,
            &cachedMoveCount, _cachedMoves.data(), _cachedPriors.data());
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
            root->terminalValue = (_position.checkers() ? CHESSCOACH_VALUE_LOSE : CHESSCOACH_VALUE_DRAW);
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
        return std::nanf("");
    }

    // Received a prediction from the network.

    // Mix in the Stockfish evaluation when available.
    // Use lc0 conversion for now (12800 = 1; for Stockfish 10k is found win, 32k is found mate).
    //
    // The important thing here is that this helps guide the MCTS search and thus the policy training,
    // but doesn't train the value head: that is still based purely on game result, so the network isn't
    // trying to learn a linear human evaluation function.
    if (!_position.checkers())
    {
        const Value stockfishValue = Eval::evaluate(_position);
        const float stockfishProbability11 = (std::atanf(static_cast<float>(stockfishValue) / 111.714640912f) / 1.5620688421f);
        const float stockfishProbability01 = std::clamp(((stockfishProbability11 + 1.f) / 2.f), 0.f, 1.f);

        // TODO: Lerp based on training progress.
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
        if (moveCount > INetwork::MaxBranchMoves)
        {
            LimitBranchingToBest(moveCount, _cachedMoves.data(), _cachedPriors.data());
            moveCount = INetwork::MaxBranchMoves;
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
    assert(moveCount > INetwork::MaxBranchMoves);

    for (int i = 0; i < INetwork::MaxBranchMoves; i++)
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
    if (Ply() < Config::NumSampingMoves)
    {
        // Use temperature=1; i.e., no need to exponentiate, just use visit counts as the distribution.
        const int sumChildVisits = _root->SumChildVisits();
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
        int maxVisitCount = std::numeric_limits<int>::min();
        std::pair<Move, Node*> maxVisited;
        for (const auto& pair : _root->children)
        {
            const int visitCount = pair.second->visitCount;
            if (visitCount > maxVisitCount)
            {
                maxVisitCount = visitCount;
                maxVisited = pair;
            }
        }
        assert(maxVisited.second);
        return maxVisited;
    }
}

void SelfPlayGame::StoreSearchStatistics()
{
    std::map<Move, float> visits;
    const int sumVisits = _root->SumChildVisits();
    for (const auto& pair : _root->children)
    {
        visits[pair.first] = static_cast<float>(pair.second->visitCount) / sumVisits;
    }
    _childVisits.emplace_back(std::move(visits));
}

StoredGame SelfPlayGame::Store() const
{
    return StoredGame(Result(), _history, _childVisits);
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
{
}

void SelfPlayWorker::ResetGames()
{
    for (int i = 0; i < Config::PredictionBatchSize; i++)
    {
        SetUpGame(i);
    }
}

void SelfPlayWorker::PlayGames(WorkCoordinator& workCoordinator, Storage* storage, INetwork* network, int maxNodesPerThread)
{
    // Initialize on the worker thread.
    Initialize(storage, maxNodesPerThread);

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
                    workCoordinator.OnWorkItemCompleted();

                    SetUpGame(i);
                    Play(i);
                }
            }

            // GPU work
            network->PredictBatch(_images.data(), _values.data(), _policies.data());
        }
    }
}

void SelfPlayWorker::Initialize(Storage* storage, int maxNodesPerThread)
{
    _storage = storage;
    Node::Allocator.Initialize(maxNodesPerThread);
}

void SelfPlayWorker::SetUpGame(int index)
{
    _states[index] = SelfPlayState::Working;
    _games[index] = SelfPlayGame(&_images[index], &_values[index], &_policies[index]);
    _gameStarts[index] = std::chrono::high_resolution_clock::now();
}

void SelfPlayWorker::DebugGame(INetwork* network, int index, const StoredGame& stored, int startingPly)
{
    SetUpGame(index);

    SelfPlayGame& game = _games[index];
    for (int i = 0; i < startingPly; i++)
    {
        game.ApplyMove(Move(stored.moves[i]));
    }

    while (true)
    {
        Play(index);
        assert(_states[index] == SelfPlayState::WaitingForPrediction);
        network->PredictBatch(_images.data(), _values.data(), _policies.data());
    }
}

void SelfPlayWorker::TrainNetwork(INetwork* network, int stepCount, int checkpoint) const
{
    for (int step = checkpoint - stepCount + 1; step <= checkpoint; step++)
    {
        auto startTrain = std::chrono::high_resolution_clock::now();

        TrainingBatch* batch = _storage->SampleBatch();
        network->TrainBatch(step, batch->images.data(), batch->values.data(), batch->policies.data());

        std::cout << "Train, step " << step << ", time " <<
            std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTrain).count() << std::endl;
    }

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
        PruneExcept(root, selected.second /* == game.Root() */);
    }

    // Take care with ordering:
    // - PruneAll() wipes anything relying on nodes, e.g. result.
    const int ply = game.Ply();
    const float result = game.Result();
    const int gameNumber = _storage->AddGame(game.Store());
    PruneAll(game.Root());

    const float gameTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - _gameStarts[index]).count();
    const float mctsTime = (gameTime / ply);
    std::cout << "Game " << gameNumber << ", ply " << ply << ", time " << gameTime << ", mcts time " << mctsTime << ", result " << result << std::endl;
    //PredictionCache::Instance.PrintDebugInfo();

    state = SelfPlayState::Finished;
}

std::pair<Move, Node*> SelfPlayWorker::RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
    std::vector<Node*>& searchPath, PredictionCacheEntry*& cacheStore)
{
    for (; mctsSimulation < Config::NumSimulations; mctsSimulation++)
    {
        if (state == SelfPlayState::Working)
        {
            if (mctsSimulation == 0)
            {
#if !DEBUG_MCTS
                AddExplorationNoise(game);
#endif

#if DEBUG_MCTS
                std::cout << "(Ready for ply " << game.Ply() << "...)" << std::endl;
                std::string _;
                std::getline(std::cin, _);
#endif
            }

            scratchGame = game;
            searchPath.clear();
            searchPath.push_back(scratchGame.Root());

            while (scratchGame.Root()->IsExpanded())
            {
                std::pair<Move, Node*> selected = SelectChild(scratchGame.Root());
                scratchGame.ApplyMoveWithRoot(selected.first, selected.second);
                searchPath.push_back(selected.second /* == scratchGame.Root() */);
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

void SelfPlayWorker::Backpropagate(const std::vector<Node*>& searchPath, float value) const
{
    // Each ply has a different player, so flip each time.
    for (Node* node : searchPath)
    {
        node->visitCount++;
        node->valueSum += value;
        value = SelfPlayGame::FlipValue(value);
    }
}

void SelfPlayWorker::PruneExcept(Node* root, Node* except) const
{
    for (auto& pair : root->children)
    {
        if (pair.second != except)
        {
            PruneAll(pair.second);
        }
    }
    delete root;
}

void SelfPlayWorker::PruneAll(Node* root) const
{
    for (auto& pair : root->children)
    {
        PruneAll(pair.second);
    }
    delete root;
}