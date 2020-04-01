#include "SelfPlay.h"

#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>

#include <Stockfish/thread.h>

#include "Config.h"

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

SelfPlayGame::SelfPlayGame(InputPlanes* image, float* value, OutputPlanes* policy)
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

float SelfPlayGame::ExpandAndEvaluate(SelfPlayState& state)
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
        if (_position.is_draw(plyToSearchRoot))
        {
            state = SelfPlayState::Working;
            root->terminalValue = CHESSCOACH_VALUE_DRAW;
            return root->terminalValue;
        }

        // Generate legal moves.
        _expandAndEvaluate_endMoves = generate<LEGAL>(_position, _expandAndEvaluate_moves);

        // Check for checkmate and stalemate.
        if (_expandAndEvaluate_moves == _expandAndEvaluate_endMoves)
        {
            state = SelfPlayState::Working;
            root->terminalValue = (_position.checkers() ? CHESSCOACH_VALUE_LOSE : CHESSCOACH_VALUE_DRAW);
            return root->terminalValue;
        }

        // Not terminal, so expand the node with legal moves.

        // Prepare for a prediction from the network.
        *_image = GenerateImage();
        state = SelfPlayState::WaitingForPrediction;
        return std::nanf("");
    }

    // Received a prediction from the network.

    // Index legal moves into the policy output planes to get logits,
    // then calculate softmax over them to get normalized probabilities for priors.
    std::vector<float> logits;
    for (ExtMove* cur = _expandAndEvaluate_moves; cur != _expandAndEvaluate_endMoves; cur++)
    {
        logits.push_back(PolicyValue(*_policy, cur->move));
    }
    std::vector<float> priors = Softmax(logits);

    // Expand child nodes with the calculated priors.
    int i = 0;
    for (const ExtMove* cur = _expandAndEvaluate_moves; cur != _expandAndEvaluate_endMoves; cur++, i++)
    {
        root->children[cur->move] = new Node(priors[i]);
    }

    state = SelfPlayState::Working;
    return *_value;
}

std::vector<float> SelfPlayGame::Softmax(const std::vector<float>& logits) const
{
    std::vector<float> probabilities(logits.size());

    const float max = *std::max_element(logits.begin(), logits.end());

    float expSum = 0.f;
    for (float logit : logits)
    {
        expSum += std::expf(logit - max);
    }

    const float logSumExp = std::logf(expSum) + max;
    for (int i = 0; i < logits.size(); i++)
    {
        probabilities[i] = std::expf(logits[i] - logSumExp);
    }

    return probabilities;
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
        return std::pair(MOVE_NULL, nullptr);
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
    std::unordered_map<Move, float> visits;
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
    , _states(INetwork::PredictionBatchSize)
    , _images(INetwork::PredictionBatchSize)
    , _values(INetwork::PredictionBatchSize)
    , _policies(INetwork::PredictionBatchSize)
    , _games(INetwork::PredictionBatchSize)
    , _scratchGames(INetwork::PredictionBatchSize)
    , _gameStarts(INetwork::PredictionBatchSize)
    , _mctsSimulations(INetwork::PredictionBatchSize, 0)
    , _searchPaths(INetwork::PredictionBatchSize)
{
}

void SelfPlayWorker::Initialize(Storage* storage)
{
    _storage = storage;
}

void SelfPlayWorker::ResetGames()
{
    for (int i = 0; i < INetwork::PredictionBatchSize; i++)
    {
        SetUpGame(i);
    }
}

void SelfPlayWorker::PlayGames(WorkCoordinator& workCoordinator, INetwork* network)
{
    // Clear away old games in progress to ensure that new ones use the new network.
    ResetGames();

    while (!workCoordinator.AllWorkItemsCompleted())
    {
        // CPU work
        for (int i = 0; i < _games.size(); i++)
        {
            Play(i);
            if (_states[i] == SelfPlayState::Finished)
            {
                workCoordinator.OnWorkItemCompleted();

                SetUpGame(i);
                Play(i);
            }
            assert(_states[i] == SelfPlayState::WaitingForPrediction);
        }
        
        // GPU work
        network->PredictBatch(_images.data(), _values.data(), _policies.data());
    }
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

        TrainingBatch batch = _storage->SampleBatch();
        network->TrainBatch(step, batch.images, batch.values, batch.policies);

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
        game.ExpandAndEvaluate(state);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }
    }

    while (!game.IsTerminal())
    {
        Node* root = game.Root();
        std::pair<Move, Node*> selected = RunMcts(game, _scratchGames[index], _states[index], _mctsSimulations[index], _searchPaths[index]);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }

        assert(selected.second != nullptr);
        game.StoreSearchStatistics();
        game.ApplyMoveWithRootAndHistory(selected.first, selected.second);
        Prune(root, selected.second /* == game.Root() */);
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

    state = SelfPlayState::Finished;
}

std::pair<Move, Node*> SelfPlayWorker::RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation, std::vector<Node*>& searchPath)
{
    for (; mctsSimulation < Config::NumSimulations; mctsSimulation++)
    {
        if (state == SelfPlayState::Working)
        {
            if (mctsSimulation == 0)
            {
                AddExplorationNoise(game);

#if DEBUG_MCTS
                std::cout << "(Ready for ply " << game.Ply() << "...)" << std::endl;
                std::string _;
                std::getline(std::cin, _);
#endif
            }

            scratchGame = game;
            searchPath = std::vector<Node*>{ scratchGame.Root() };

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

        // The value we get is from the final node of the scratch game (could be WHITE or BLACK)
        // and we start applying it at the current position of the actual game (could again be WHITE or BLACK),
        // so flip it if they differ.
        float value = scratchGame.ExpandAndEvaluate(state);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return std::pair(MOVE_NULL, nullptr);
        }

        // Always value a node from the parent's to-play perspective.
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

void SelfPlayWorker::Prune(Node* root, Node* except) const
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