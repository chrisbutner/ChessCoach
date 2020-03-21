#include "SelfPlay.h"

#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>

#include <Stockfish/thread.h>

int Config::NumSampingMoves = 30;
int Config::MaxMoves = 512;
int Config::NumSimulations = 800;

float Config::RootDirichletAlpha = 0.3f;
float Config::RootExplorationFraction = 0.25f;

float Config::PbCBase = 19652.f;
float Config::PbCInit = 1.25f;

std::default_random_engine Config::Random;

Node::Node(float setPrior)
    : originalPrior(setPrior)
    , prior(setPrior)
    , visitCount(0)
    , valueSum(0.f)
    , terminalValue(CHESSCOACH_VALUE_UNINITIALIZED)
    , _sumChildVisits(0)
{
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

Game::Game()
    : _positionStates(new std::deque<StateInfo>(1))
    , _root(new Node(0.f))
{
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    _position.set(StartFEN, false /* isChess960 */, &_positionStates->back(), Threads.main());
}

Game::Game(const Game& other)
    : _position(other._position)
    , _positionStates(new std::deque<StateInfo>())
    , _root(other._root)
{
    assert(&other != this);
}

Game& Game::operator=(const Game& other)
{
    assert(&other != this);

    _position = other._position;
    _positionStates.reset(new std::deque<StateInfo>());
    _root = other._root;

    return *this;
}

Game::~Game()
{
}

Node* Game::Root() const
{
    return _root;
}

bool Game::IsTerminal() const
{
    return (_root->terminalValue != CHESSCOACH_VALUE_UNINITIALIZED) || (Ply() >= Config::MaxMoves);
}

Color Game::ToPlay() const
{
    return _position.side_to_move();
}

void Game::ApplyMove(Move move, Node* newRoot)
{
    _positionStates->emplace_back();
    _position.do_move(move, _positionStates->back());
    _root = newRoot;
}

float Game::ExpandAndEvaluate(INetwork* network)
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
    InputPlanes image = MakeImage();
    const IPrediction* prediction = network->Predict(image);
    const float value = prediction->Value();
    const OutputPlanesPtr policy = reinterpret_cast<float(*)[8][8]>(prediction->Policy());

    // Index legal moves into the policy output planes to get logits,
    // then calculate softmax over them to get normalized probabilities for priors.
    std::vector<float> logits;
    for (ExtMove* cur = moves; cur != endMoves; cur++)
    {
        logits.push_back(GetLogit(policy, cur->move));
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

std::vector<float> Game::Softmax(const std::vector<float>& logits) const
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

float Game::GetLogit(const OutputPlanesPtr policy, Move move) const
{
    // TODO: Still need to port Python->C++, python-chess->stockfish
    return 0.f;
}

InputPlanes Game::MakeImage() const
{
    // TODO: Still need to port Python->C++, python-chess->stockfish
    return InputPlanes();
}

std::pair<Move, Node*> Game::SelectMove() const
{
    if (Ply() < Config::NumSampingMoves)
    {
        // Use temperature=1; i.e., no need to exponentiate, just use visit counts as the distribution.
        const int sumChildVisits = _root->SumChildVisits();
        int sample = std::uniform_int_distribution<int>(0, sumChildVisits - 1)(Config::Random);
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

void Game::StoreSearchStatistics()
{
    std::unordered_map<Move, float> visits;
    const int sumVisits = _root->SumChildVisits();
    for (auto pair : _root->children)
    {
        visits[pair.first] = static_cast<float>(pair.second->visitCount) / sumVisits;
    }
    _childVisits.emplace_back(std::move(visits));
}

int Game::Ply() const
{
    return _position.game_ply();
}

void Mcts::Work(INetwork* network) const
{
    while (true)
    {
        Play(network);
    }
}

void Mcts::Play(INetwork* network) const
{
    auto startGame = std::chrono::high_resolution_clock::now();

    Game game;
    game.ExpandAndEvaluate(network);

    while (!game.IsTerminal())
    {
        auto startMcts = std::chrono::high_resolution_clock::now();
        Node* root = game.Root();
        std::pair<Move, Node*> selected = RunMcts(network, game);
        game.StoreSearchStatistics();
        game.ApplyMove(selected.first, selected.second);
        Prune(root, selected.second /* == game.Root() */); // Can't instead auto-trigger by ApplyMove because of scratch games, which share nodes.
        std::cout << "MCTS, ply " << game.Ply() << ", time " <<
            std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startMcts).count() << std::endl;
    }

    // TODO: Gather results
    PruneAll(game.Root()); // Can't instead auto-trigger by destructor because of scratch games, which share nodes.

    std::cout << "Game, ply " << game.Ply() << ", time " <<
        std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startGame).count() << std::endl;
}

std::pair<Move, Node*> Mcts::RunMcts(INetwork* network, Game& game) const
{
    AddExplorationNoise(game);

    for (int i = 0; i < Config::NumSimulations; i++)
    {
        Game scratchGame = game;
        std::vector<Node*> searchPath{ scratchGame.Root() };

        while (scratchGame.Root()->IsExpanded())
        {
            std::pair<Move, Node*> selected = SelectChild(scratchGame.Root());
            scratchGame.ApplyMove(selected.first, selected.second);
            searchPath.push_back(selected.second /* == scratchGame.Root() */);
        }

        // The value we get is from the final node of the scratch game (could be WHITE or BLACK)
        // and we start applying it at the current position of the actual game (could again be WHITE or BLACK),
        // so flip it if they differ.
        float value = scratchGame.ExpandAndEvaluate(network);
        value = FlipValue(Color(game.ToPlay() ^ scratchGame.ToPlay()), value);
        Backpropagate(searchPath, value);
    }

    return game.SelectMove();
}

void Mcts::AddExplorationNoise(Game& game) const
{
    std::gamma_distribution<float> noise(Config::RootDirichletAlpha, 1.f);
    for (auto pair : game.Root()->children)
    {
        Node* child = pair.second;
        const float childNoise = noise(Config::Random);
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
        value = FlipValue(value);
    }
}

float Mcts::FlipValue(Color toPlay, float value) const
{
    return (toPlay == WHITE) ? value : FlipValue(value);
}

float Mcts::FlipValue(float value) const
{
    return (CHESSCOACH_VALUE_WIN - value);
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