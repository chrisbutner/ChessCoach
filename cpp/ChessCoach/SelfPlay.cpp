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
        for (auto pair : children)
        {
            _sumChildVisits += pair.second->visitCount;
        }
        assert(_sumChildVisits > 0);
    }
    return _sumChildVisits;
}

// TODO: Write a custom allocator for nodes (work out very maximum, then do some kind of ring/tree - important thing is all same size, capped number)
// TODO: Also input/output planes? e.g. for StoredGames vector storage

SelfPlayGame::SelfPlayGame()
    : Game()
    , _root(new Node(0.f))
{
}

SelfPlayGame::SelfPlayGame(const SelfPlayGame& other)
    : Game(other)
    , _root(other._root)
{
    assert(&other != this);
}

SelfPlayGame& SelfPlayGame::operator=(const SelfPlayGame& other)
{
    assert(&other != this);

    Game::operator=(other);

    _root = other._root;

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

float SelfPlayGame::ExpandAndEvaluate(INetwork* network)
{
    Node* root = _root;
    assert(!root->IsExpanded());

    // A known-terminal leaf will remain a leaf, so be prepared to
    // quickly return its terminal value on repeated visits.
    if (root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED)
    {
        return root->terminalValue;
    }

    // Check for draw by 50-move or 3-repetition.
    if (_position.is_draw(Ply()))
    {
        root->terminalValue = CHESSCOACH_VALUE_DRAW;
        return root->terminalValue;
    }

    // Generate legal moves.
    ExtMove moves[MAX_MOVES];
    ExtMove* endMoves = generate<LEGAL>(_position, moves);

    // Check for checkmate and stalemate.
    if (moves == endMoves)
    {
        root->terminalValue = (_position.checkers() ? CHESSCOACH_VALUE_LOSE : CHESSCOACH_VALUE_DRAW);
        return root->terminalValue;
    }

    // Not terminal, so expand the node with legal moves.

    // Get a prediction from the network - either the uniform policy or NN policy.
    InputPlanes image = GenerateImage();
    std::unique_ptr<IPrediction> prediction(network->Predict(image));
    if (!prediction)
    {
        // Terminating self-play.
        return std::nanf("");
    }
    const float value = prediction->Value();
    const OutputPlanesPtr policy = reinterpret_cast<float(*)[8][8]>(prediction->Policy());

    // Index legal moves into the policy output planes to get logits,
    // then calculate softmax over them to get normalized probabilities for priors.
    std::vector<float> logits;
    for (ExtMove* cur = moves; cur != endMoves; cur++)
    {
        logits.push_back(PolicyValue(policy, cur->move));
    }
    std::vector<float> priors = Softmax(logits);

    // Expand child nodes with the calculated priors.
    int i = 0;
    for (const ExtMove* cur = moves; cur != endMoves; cur++, i++)
    {
        root->children[cur->move] = new Node(priors[i]);
    }

    return value;
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
        for (auto pair : _root->children)
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
        for (auto pair : _root->children)
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
    for (auto pair : _root->children)
    {
        visits[pair.first] = static_cast<float>(pair.second->visitCount) / sumVisits;
    }
    _childVisits.emplace_back(std::move(visits));
}

// Store() leaves the Game in an inconsistent state, w.r.t. _position vs. _positionStates vs. _history.
// This is intentional laziness to avoid unnecessary housekeeping, since the Game will no longer be needed.
StoredGame SelfPlayGame::Store()
{
    return StoredGame(Result(), _history, _childVisits);
}

Mcts::Mcts(Storage* storage)
    : _storage(storage)
{
}

void Mcts::PlayGames(WorkCoordinator& workCoordinator, INetwork* network) const
{
    while (Play(network))
    {
        workCoordinator.OnWorkItemCompleted();
    }
}

void Mcts::TrainNetwork(INetwork* network, int stepCount, int checkpoint) const
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

bool Mcts::Play(INetwork* network) const
{
    auto startGame = std::chrono::high_resolution_clock::now();

    SelfPlayGame game;
    float check = game.ExpandAndEvaluate(network);
    if (std::isnan(check))
    {
        // Terminating self-play.
        return false;
    }

    while (!game.IsTerminal())
    {
        //auto startMcts = std::chrono::high_resolution_clock::now();
        Node* root = game.Root();
        std::pair<Move, Node*> selected = RunMcts(network, game);
        if (selected.second == nullptr)
        {
            // Terminating self-play.
            return false;
        }
        game.StoreSearchStatistics();
        game.ApplyMoveWithRootAndHistory(selected.first, selected.second);
        Prune(root, selected.second /* == game.Root() */);
        //std::cout << "MCTS, ply " << game.Ply() << ", time " <<
        //    std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startMcts).count() << std::endl;
    }

    // Take care with ordering:
    // - Store() wipes anything relying on the position, e.g. ply.
    // - PruneAll() wipes anything relying on nodes, e.g. terminal value.
    const int ply = game.Ply();
    const float result = game.Result();
    const int gameNumber = _storage->AddGame(game.Store());
    PruneAll(game.Root());

    const float gameTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startGame).count();
    const float mctsTime = (gameTime / ply);
    std::cout << "Game " << gameNumber << ", ply " << ply << ", time " << gameTime << ", mcts time " << mctsTime << ", result " << result << std::endl;

    return true;
}

std::pair<Move, Node*> Mcts::RunMcts(INetwork* network, SelfPlayGame& game) const
{
    AddExplorationNoise(game);

    for (int i = 0; i < Config::NumSimulations; i++)
    {
        SelfPlayGame scratchGame = game;
        std::vector<Node*> searchPath{ scratchGame.Root() };

        while (scratchGame.Root()->IsExpanded())
        {
            std::pair<Move, Node*> selected = SelectChild(scratchGame.Root());
            scratchGame.ApplyMoveWithRoot(selected.first, selected.second);
            searchPath.push_back(selected.second /* == scratchGame.Root() */);
        }

        // The value we get is from the final node of the scratch game (could be WHITE or BLACK)
        // and we start applying it at the current position of the actual game (could again be WHITE or BLACK),
        // so flip it if they differ.
        float value = scratchGame.ExpandAndEvaluate(network);
        if (std::isnan(value))
        {
            // Terminating self-play.
            return std::pair(MOVE_NULL, nullptr);
        }
        value = SelfPlayGame::FlipValue(Color(game.ToPlay() ^ scratchGame.ToPlay()), value);
        Backpropagate(searchPath, value);
    }

    return game.SelectMove();
}

void Mcts::AddExplorationNoise(SelfPlayGame& game) const
{
    std::gamma_distribution<float> noise(Config::RootDirichletAlpha, 1.f);
    for (auto pair : game.Root()->children)
    {
        Node* child = pair.second;
        const float childNoise = noise(SelfPlayGame::Random);
        child->prior = child->originalPrior * (1 - Config::RootExplorationFraction) + childNoise * Config::RootExplorationFraction;
    }
}

std::pair<Move, Node*> Mcts::SelectChild(const Node* parent) const
{
    float maxUcbScore = -std::numeric_limits<float>::infinity();
    std::pair<Move, Node*> max;
    for (auto pair : parent->children)
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
float Mcts::CalculateUcbScore(const Node* parent, const Node* child) const
{
    const float pbC = (std::logf((parent->visitCount + Config::PbCBase + 1.f) / Config::PbCBase) + Config::PbCInit) *
        std::sqrtf(static_cast<float>(parent->visitCount)) / (child->visitCount + 1.f);
    const float priorScore = pbC * child->prior;
    return priorScore + child->Value();
}

void Mcts::Backpropagate(const std::vector<Node*>& searchPath, float value) const
{
    // Each ply has a different player, so flip each time.
    for (Node* node : searchPath)
    {
        node->visitCount++;
        node->valueSum += value;
        value = SelfPlayGame::FlipValue(value);
    }
}

void Mcts::Prune(Node* root, Node* except) const
{
    for (auto pair : root->children)
    {
        if (pair.second != except)
        {
            PruneAll(pair.second);
        }
    }
    delete root;
}

void Mcts::PruneAll(Node* root) const
{
    for (auto pair : root->children)
    {
        PruneAll(pair.second);
    }
    delete root;
}