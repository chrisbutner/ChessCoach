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
#include "Pgn.h"
#include "Random.h"



TerminalValue TerminalValue::NonTerminal()
{
    return {};
}

int TerminalValue::Draw()
{
    return 0;
}

// Mate in N fullmoves, not halfmoves/ply.
int TerminalValue::MateIn(int n)
{
    return n;
}

// Opponent mate in N fullmoves, not halfmoves/ply.
int TerminalValue::OpponentMateIn(int n)
{
    return -n;
}

TerminalValue::TerminalValue()
    : _value()
    , _mateTerm(0.f)
{
}

TerminalValue::TerminalValue(const int value)
{
    operator=(value);
}

TerminalValue& TerminalValue::operator=(const int value)
{
    _value = value;

    if (value > 0)
    {
        const int mateNSaturated = std::min(static_cast<int>(Game::UcbMateTerm.size() - 1), value);
        _mateTerm = Game::UcbMateTerm[mateNSaturated];
    }
    else
    {
        // No adjustment for opponent-mate-in-N. The goal of the search in that situation is already
        // to go wide rather than deep and find some paths with value. Adding disincentives (with some
        // variation of inverse exploration rate coefficient) can help exhaustive searches finish in
        // fewer nodes in opponent-mate-in-N trees; however, the calculations slow down the search to
        // more processing time overall despite fewer nodes, and worse principle variations are preferred
        // before the exhaustive search finishes, because better priors get searched and disincentivized
        // sooner. So, rely on every-other-step mate-in-N incentives to help guide search, and SelectMove
        // preferring slower opponent mates (in the worst case).
        //
        // Also, no adjustment for draws at the moment.
        _mateTerm = 0.f;
    }

    return *this;
}

bool TerminalValue::operator==(const int other) const
{
    return (_value == other);
}

bool TerminalValue::IsNonTerminal() const
{
    return !_value;
}

bool TerminalValue::IsImmediate() const
{
    return (_value &&
        ((*_value == Draw()) || (*_value == MateIn<1>())));
}

float TerminalValue::ImmediateValue() const
{
    // Coalesce a draw for the (Ply >= MaxMoves) and other undetermined/unfinished cases.
    if (_value == TerminalValue::MateIn<1>())
    {
        return CHESSCOACH_VALUE_WIN;
    }
    return CHESSCOACH_VALUE_DRAW;
}

bool TerminalValue::IsMateInN() const
{
    return (_value && (*_value > 0));
}

bool TerminalValue::IsOpponentMateInN() const
{
    return (_value && (*_value < 0));
}

int TerminalValue::MateN() const
{
    return (_value ? std::max(0, *_value) : 0);
}

int TerminalValue::OpponentMateN() const
{
    return (_value ? std::max(0, -*_value) : 0);
}

int TerminalValue::EitherMateN() const
{
    return (_value ? *_value : 0);
}

float TerminalValue::MateScore(float explorationRate) const
{
    return (explorationRate * _mateTerm);
}

thread_local PoolAllocator<Node, Node::BlockSizeBytes> Node::Allocator;

Node::Node(Move setMove, float setPrior)
    : move(setMove)
    , prior(setPrior)
    , visitCount(0)
    , visitingCount(0)
    , valueSum(0.f)
    , terminalValue{}
    , expanding(false)
    , bestChild(nullptr)
    , firstChild(nullptr)
    , nextSibling(nullptr)
{
    assert(!std::isnan(setPrior));
}

void* Node::operator new(size_t /*count*/)
{
    return Allocator.Allocate();
}

void Node::operator delete(void* ptr) noexcept
{
    Allocator.Free(ptr);
}

Node::iterator Node::begin()
{
    return iterator(firstChild);
}

Node::iterator Node::end()
{
    return iterator(nullptr);
}

Node::const_iterator Node::begin() const
{
    return const_iterator(firstChild);
}

Node::const_iterator Node::end() const
{
    return const_iterator(nullptr);
}

Node::const_iterator Node::cbegin() const
{
    return const_iterator(firstChild);
}

Node::const_iterator Node::cend() const
{
    return const_iterator(nullptr);
}

bool Node::IsExpanded() const
{
    return (firstChild != nullptr);
}

float Node::Value() const
{
    // First-play urgency (FPU) is zero, a loss.
    if (visitCount <= 0)
    {
        return CHESSCOACH_VALUE_LOSS;
    }

    return (valueSum / visitCount);
}

Node* Node::Child(Move match)
{
    for (Node& child : *this)
    {
        if (child.move == match)
        {
            return &child;
        }
    }
    return nullptr;
}

int Node::CountChildren() const
{
    return std::distance(begin(), end());
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
    , _searchRootPly(Ply())
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
    : Game()
    , _root(new Node(MOVE_NONE, 0.f))
    , _tryHard(false)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _searchRootPly(Ply())
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(const std::string& fen, const std::vector<Move>& moves, bool tryHard,
    INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
    : Game(fen, moves)
    , _root(new Node(MOVE_NONE, 0.f))
    , _tryHard(tryHard)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _searchRootPly(Ply()) // Important for this to be moves.size() when searching positions.
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
    , _mctsValues(std::move(other._mctsValues))
    , _childVisits(std::move(other._childVisits))
    , _history(std::move(other._history))
    , _result(other._result)
{
    assert(&other != this);

    other._root = nullptr;
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
    _mctsValues = std::move(other._mctsValues);
    _childVisits = std::move(other._childVisits);
    _history = std::move(other._history);
    _result = other._result;

    other._root = nullptr;

    return *this;
}

SelfPlayGame::~SelfPlayGame()
{
}

SelfPlayGame SelfPlayGame::SpawnShadow(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy)
{
    SelfPlayGame shadow(*this);

    shadow._image = image;
    shadow._value = value;
    shadow._policy = policy;

    return shadow;
}

Node* SelfPlayGame::Root() const
{
    return _root;
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
    //
    // Sum child visits and assign. For most nodes we could just decrement because the node was visited
    // exactly once as a leaf before being expanded. However, 2-repetition draws complicate things because
    // they could be visited multiple times as a terminal draw, then become non-terminal as the game progresses
    // past the first occurence of the position and get expanded. Pack the terminal visit count into the Node
    // to speed this up if required.
    //
    // We also need to adjust the value sum down to match the smaller visit count, so that MCTS value
    // stays accurate and in the [0, 1] range. It would be best to excise the original as-leaf neural
    // network evaluation, but without adding additional Node fields/tracking that's already mixed in,
    // so just average down to the new visit count.
    const int previousVisitCount = _root->visitCount;

    _root->visitCount = 0;
    for (const Node& child : *_root)
    {
        _root->visitCount += child.visitCount;
    }

    if (previousVisitCount > 0)
    {
        _root->valueSum *= (static_cast<float>(_root->visitCount) / previousVisitCount);
    }
}

float SelfPlayGame::ExpandAndEvaluate(SelfPlayState& state, PredictionCacheChunk*& cacheStore)
{
    Node* root = _root;
    assert(!root->IsExpanded());

    // A known-terminal leaf will remain a leaf, so be prepared to
    // quickly return its terminal value on repeated visits.
    if (root->terminalValue.IsImmediate())
    {
        state = SelfPlayState::Working;
        return root->terminalValue.ImmediateValue();
    }

    // It's very important in this method to always value a node from the parent's to-play perspective, so:
    // - flip network evaluations
    // - value checkmate as a win
    //
    // This seems a little counter-intuitive, but it's an artifact of storing priors/values/visits on
    // the child nodes themselves instead of on "edges" belonging to the parent.
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

    if (state == SelfPlayState::Working)
    {
        // Generate legal moves.
        _expandAndEvaluate_endMoves = generate<LEGAL>(_position, _expandAndEvaluate_moves);

        // Check for checkmate and stalemate.
        const int workingMoveCount = static_cast<int>(_expandAndEvaluate_endMoves - _expandAndEvaluate_moves);
        if (workingMoveCount == 0)
        {
            // Value from the parent's perspective.
            root->terminalValue = (_position.checkers() ? TerminalValue::MateIn<1>() : TerminalValue::Draw());
            return root->terminalValue.ImmediateValue();
        }

        // Check for draw by 50-move or repetition.
        // This has to happen after checking for mate in order to follow game rules.
        //
        // Use slightly confusing terminology:
        // - a 1-repetition means the first occurence (actually 0 repetitions)
        // - a 2-repetition means the second occurence (actually 1 repetition)
        // - a 3-repetition means the third occurence (actually 2 repetitions)
        //
        // Stockfish checks for (a) two-fold repetition strictly after the search root
        // (e.g. search-root, rep-1, rep-2) or (b) three-fold repetition anywhere
        // (e.g. rep-1, rep-2, search-root, rep-3) in order to terminate and prune efficiently.
        //
        // We can use the same logic safely because we're path-dependent: no post-search valuations
        // are hashed purely by position (only network-dependent predictions, potentially),
        // and nodes with identical positions reached differently are distinct in the tree
        // This saves time in the 800-simulation budget for more useful exploration.
        //
        // However, once the search root advances, reusing a sub-tree requires reevaluating any
        // previous 2-repetitions that may no longer be so (because their earlier occurence
        // got played), including possibly the root itself. So, for 2-repetitions, short-circuit here and
        // return a draw score but don't cache it on the node. Instead, check again next time.
        if (IsDrawByNoProgressOrThreefoldRepetition())
        {
            // Value from the parent's perspective (easy, it's a draw).
            root->terminalValue = TerminalValue::Draw();
            return root->terminalValue.ImmediateValue();
        }
        const int plyToSearchRoot = (Ply() - _searchRootPly);
        if (IsDrawByTwofoldRepetition(plyToSearchRoot))
        {
            // Value from the parent's perspective (easy, it's a draw).
            // Don't cache the 2-repetition; check again next time.
            assert(root->terminalValue.IsNonTerminal());
            return TerminalValue(TerminalValue::Draw()).ImmediateValue();
        }

        // Try get a cached prediction. Only hit the cache up to a max ply for self-play since we
        // see enough unique positions/paths to fill the cache no matter what, and it saves on time
        // to evict less. However, in search (TryHard) it's better to keep everything recent.
        cacheStore = nullptr;
        float cachedValue = std::numeric_limits<float>::quiet_NaN();
        _imageKey = GenerateImageKey();
        bool hitCached = false;
        if ((workingMoveCount <= PredictionCacheEntry::MaxMoveCount) &&
            (TryHard() || (Ply() <= Config::Misc.PredictionCache_MaxPly)))
        {
            hitCached = PredictionCache::Instance.TryGetPrediction(_imageKey, workingMoveCount,
                &cacheStore, &cachedValue, _cachedPriors.data());
        }
        if (hitCached)
        {
            // Expand child nodes with the cached priors.
            Expand(workingMoveCount);
            return cachedValue;
        }

        // Prepare for a prediction from the network.
        GenerateImage(*_image);
        state = SelfPlayState::WaitingForPrediction;
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Received a prediction from the network.

    // Value from the parent's perspective.
    const float value = FlipValue(*_value);

    // Index legal moves into the policy output planes to get logits,
    // then calculate softmax over them to get normalized probabilities for priors.
    int moveCount = 0;
    for (ExtMove* cur = _expandAndEvaluate_moves; cur != _expandAndEvaluate_endMoves; cur++)
    {
        _cachedPriors[moveCount] = PolicyValue(*_policy, cur->move); // Logits
        moveCount++;
    }
    Softmax(moveCount, _cachedPriors.data()); // Logits -> priors

    // Expand child nodes with the calculated priors.
    Expand(moveCount);

    // Store in the cache if appropriate.
    if (cacheStore)
    {
        assert(moveCount <= PredictionCacheEntry::MaxMoveCount);
        cacheStore->Put(_imageKey, value, moveCount, _cachedPriors.data());
    }

    state = SelfPlayState::Working;
    return value;
}

void SelfPlayGame::Expand(int moveCount)
{
    Node* root = _root;
    assert(!root->IsExpanded());
    assert(root->firstChild == nullptr);
    assert(moveCount > 0);

    assert(_position.legal(_expandAndEvaluate_moves[0].move));
    Node* lastSibling = new Node(_expandAndEvaluate_moves[0].move, _cachedPriors[0]);
    root->firstChild = lastSibling;
    for (int i = 1; i < moveCount; i++)
    {
        const Move move = _expandAndEvaluate_moves[i].move;
        assert(_position.legal(move));
        lastSibling->nextSibling = new Node(move, _cachedPriors[i]);
        lastSibling = lastSibling->nextSibling;
    }
}

// Avoid Position::is_draw because it regenerates legal moves.
// If we've already just checked for checkmate and stalemate then this works fine.
bool SelfPlayGame::IsDrawByNoProgressOrThreefoldRepetition()
{
    const StateInfo* stateInfo = _position.state_info();

    return
        // Omit "and not checkmate" from Position::is_draw.
        (stateInfo->rule50 > 99) ||
        // Stockfish encodes 3-repetition as negative.
        (stateInfo->repetition < 0);
}

// Avoid Position::is_draw because it regenerates legal moves.
// If we've already just checked for checkmate and stalemate then this works fine.
bool SelfPlayGame::IsDrawByTwofoldRepetition(int plyToSearchRoot)
{
    const StateInfo* stateInfo = _position.state_info();

    // Return a draw score if a position repeats once earlier but strictly
    // after the root, or repeats twice before or at the root.
    //
    // Check for >0 rather than non-zero to exclude 3-repetition.
    return ((stateInfo->repetition > 0) && (stateInfo->repetition < plyToSearchRoot));
}

void SelfPlayGame::Softmax(int moveCount, float* distribution) const
{
    const float max = *std::max_element(distribution, distribution + moveCount);

    float expSum = 0.f;
    for (int i = 0; i < moveCount; i++)
    {
        expSum += ::expf(distribution[i] - max);
    }

    const float logSumExp = ::logf(expSum) + max;
    for (int i = 0; i < moveCount; i++)
    {
        distribution[i] = ::expf(distribution[i] - logSumExp);
    }
}

void SelfPlayGame::StoreSearchStatistics()
{
    std::map<Move, float> visits;
    const int sumChildVisits = _root->visitCount;
    for (const Node& child : *_root)
    {
        visits[child.move] = static_cast<float>(child.visitCount) / sumChildVisits;
    }
    // Flip to the side to play's perspective (NOT the parent's perspective, like in the actual MCTS tree).
    _mctsValues.push_back(FlipValue(_root->Value()));
    _childVisits.emplace_back(std::move(visits));
}

void SelfPlayGame::Complete()
{
    // Save state that depends on nodes.
    // Terminal value is from the parent's perspective, so unconditionally flip (~)
    // from *parent* to *self* before flipping from ToPlay() to white's perspective.
    _result = FlipValue(~ToPlay(), _root->terminalValue.ImmediateValue());

    // Clear and detach from all nodes.
    PruneAll();
}

SavedGame SelfPlayGame::Save() const
{
    return SavedGame(Result(), _history, _mctsValues, _childVisits);
}

void SelfPlayGame::PruneExcept(Node* root, Node*& except)
{
    if (!root)
    {
        return;
    }

    // Rely on caller to already have updated the _root to the preserved subtree.
    assert(_root != root);
    assert(_root == except);

    // Minimize fragmentation by cloning "except" and pruning the original.
    _root = new Node(*except);
    _root->nextSibling = nullptr;

    // Don't let "except"'s descendants get pruned when the original is deleted.
    except->firstChild = nullptr;

    // Prune, then update the caller's "except" pointer (now deleted) to the clone.
    PruneAllInternal(root);
    except = _root;
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

// Delete siblings contiguously. For now they'll still end up reversed in the PoolAllocator
// but if benchmarking ends up better later, a "recycling" staging area can be used to correct.
void SelfPlayGame::PruneAllInternal(Node* node)
{
    Node* parent = node;
    while (parent)
    {
        if (parent->firstChild) PruneAllInternal(parent->firstChild);
        parent = parent->nextSibling;
    }
    while (node)
    {
        Node* prune = node;
        node = node->nextSibling;
        delete prune;
    }
}

void SelfPlayGame::UpdateSearchRootPly()
{
    _searchRootPly = Ply();
}

Move SelfPlayGame::ParseSan(const std::string& san)
{
    return Pgn::ParseSan(_position, san);
}

// Don't clear the prediction cache more than once every 30 seconds
// (aimed at preventing N self-play worker threads from each clearing).
Throttle SelfPlayWorker::PredictionCacheResetThrottle(30 * 1000 /* durationMilliseconds */);

SelfPlayWorker::SelfPlayWorker(const NetworkConfig& networkConfig, Storage* storage)
    : _networkConfig(&networkConfig)
    , _storage(storage)
    , _explorationRateBase(networkConfig.SelfPlay.ExplorationRateBase)
    , _explorationRateInit(networkConfig.SelfPlay.ExplorationRateInit)
    , _states(networkConfig.SelfPlay.PredictionBatchSize)
    , _images(networkConfig.SelfPlay.PredictionBatchSize)
    , _values(networkConfig.SelfPlay.PredictionBatchSize)
    , _policies(networkConfig.SelfPlay.PredictionBatchSize)
    , _games(networkConfig.SelfPlay.PredictionBatchSize)
    , _scratchGames(networkConfig.SelfPlay.PredictionBatchSize)
    , _gameStarts(networkConfig.SelfPlay.PredictionBatchSize)
    , _mctsSimulations(networkConfig.SelfPlay.PredictionBatchSize, 0)
    , _searchPaths(networkConfig.SelfPlay.PredictionBatchSize)
    , _cacheStores(networkConfig.SelfPlay.PredictionBatchSize)
    , _searchConfig{}
    , _searchState{}
{
}

const NetworkConfig& SelfPlayWorker::Config() const
{
    return *_networkConfig;
}

void SelfPlayWorker::PlayGames(WorkCoordinator& workCoordinator, INetwork* network)
{
    // Use the faster student network for self-play.
    const NetworkType networkType = NetworkType_Student;

    while (true)
    {
        // Wait until games are required.
        workCoordinator.WaitForWorkItems();

        // Generate uniform predictions for the first network (rather than use random weights).
        const bool uniform = workCoordinator.GenerateUniformPredictions();
        if (uniform)
        {
            std::cout << "Generating uniform network predictions until trained" << std::endl;
        }

        // Warm up the GIL and predictions.
        const PredictionStatus warmupStatus = WarmUpPredictions(network, networkType, 1);
        if ((warmupStatus & PredictionStatus_UpdatedNetwork) && PredictionCacheResetThrottle.TryFire())
        {
            // This thread has permission to clear the prediction cache after seeing an updated network.
            PredictionCache::Instance.Clear();
        }

        // Set up any uninitialized games. It's important to do this here so that "_gameStarts" is accurate for MCTS timing.
        // Otherwise, continue games in progress, clearing the prediction cache when the network is updated.
        for (int i = 0; i < _games.size(); i++)
        {
            if (!_games[i].Root())
            {
                SetUpGame(i);
            }
        }

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
                    SaveToStorageAndLog(network, i);

                    workCoordinator.OnWorkItemCompleted();

                    SetUpGame(i);
                    Play(i);
                }
            }

            // GPU work
            if (uniform)
            {
                PredictBatchUniform(_networkConfig->SelfPlay.PredictionBatchSize, _images.data(), _values.data(), _policies.data());
            }
            else
            {
                const PredictionStatus status = network->PredictBatch(networkType, _networkConfig->SelfPlay.PredictionBatchSize, _images.data(), _values.data(), _policies.data());
                if ((status & PredictionStatus_UpdatedNetwork) && PredictionCacheResetThrottle.TryFire())
                {
                    // This thread has permission to clear the prediction cache after seeing an updated network.
                    PredictionCache::Instance.Clear();
                }
            }
        }

        // Don't free nodes here. Let the games and MCTS trees be continued using the next network
        // after clearing the prediction cache (via PredictionStatus flag).
    }
}

void SelfPlayWorker::ClearGame(int index)
{
    _states[index] = SelfPlayState::Working;
    _gameStarts[index] = std::chrono::high_resolution_clock::now();
    _mctsSimulations[index] = 0;
    _searchPaths[index].clear();
    _cacheStores[index] = nullptr;
}

void SelfPlayWorker::SetUpGame(int index)
{
    ClearGame(index);
    _games[index] = SelfPlayGame(&_images[index], &_values[index], &_policies[index]);
}

void SelfPlayWorker::SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard)
{
    ClearGame(index);
    _games[index] = SelfPlayGame(fen, moves, tryHard, &_images[index], &_values[index], &_policies[index]);
}

void SelfPlayWorker::SetUpGameExisting(int index, const std::vector<Move>& moves, int applyNewMovesOffset)
{
    ClearGame(index);

    SelfPlayGame& game = _games[index];

    for (int i = applyNewMovesOffset; i < moves.size(); i++)
    {
        const Move move = moves[i];
        Node* root = game.Root();

        // The root may be null after taking the "else" branch below (child not explored)
        // on a previous iteration.
        Node* newRoot = nullptr;
        if (root)
        {
            for (Node& child : *root)
            {
                if (move == child.move)
                {
                    newRoot = &child;
                    break;
                }
            }
        }

        if (newRoot)
        {
            // Preserve the existing sub-tree.
            game.ApplyMoveWithRoot(move, newRoot);
            game.PruneExcept(root, newRoot);
        }
        else
        {
            newRoot = ((i == (moves.size() - 1)) ? new Node(MOVE_NONE, 0.f) : nullptr);
            game.PruneAll();
            game.ApplyMoveWithRoot(move, newRoot);
        }
    }

    // The additional moves are being applied to an existing game for efficiency, but really we're setting up
    // a new position, so update the search root ply for draw-checking.
    game.UpdateSearchRootPly();
}

void SelfPlayWorker::TrainNetwork(INetwork* network, NetworkType networkType, std::vector<GameType>& gameTypes,
    std::vector<Window>& trainingWindows, int step, int checkpoint)
{
    // Delegate to Python.
    std::cout << "Training steps " << step << "-" << checkpoint << "..." << std::endl;
    auto startTrain = std::chrono::high_resolution_clock::now();
    network->Train(networkType, gameTypes, trainingWindows, step, checkpoint);
    const float trainTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTrain).count();
    const int stepCount = (checkpoint - step + 1);
    const float trainTimePerStep = (trainTime / stepCount);
    std::cout << "Trained steps " << step << "-" << checkpoint << ", total time " << trainTime << ", step time " << trainTimePerStep << std::endl;
}

void SelfPlayWorker::TrainNetworkWithCommentary(INetwork* network, int step, int checkpoint)
{
    // Delegate to Python.
    std::cout << "Training commentary steps " << step << "-" << checkpoint << "..." << std::endl;
    auto startTrain = std::chrono::high_resolution_clock::now();
    network->TrainCommentary(step, checkpoint);
    const float trainTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTrain).count();
    const int stepCount = (checkpoint - step + 1);
    const float trainTimePerStep = (trainTime / stepCount);
    std::cout << "Trained commentary steps " << step << "-" << checkpoint << ", total time " << trainTime << ", step time " << trainTimePerStep << std::endl;
}

void SelfPlayWorker::SaveNetwork(INetwork* network, NetworkType networkType, int checkpoint)
{
    // Save the network to (a) allow stopping and resuming, and (b) give the prediction network the latest weights.
    network->SaveNetwork(networkType, checkpoint);
}

bool SelfPlayWorker::StrengthTestNetwork(INetwork* network, NetworkType networkType, int checkpoint)
{
    // Strength-test the engine every "StrengthTestInterval" steps.
    assert(_networkConfig->Training.StrengthTestInterval >= _networkConfig->Training.CheckpointInterval);
    assert((_networkConfig->Training.StrengthTestInterval % _networkConfig->Training.CheckpointInterval) == 0);
    if ((checkpoint % _networkConfig->Training.StrengthTestInterval) == 0)
    {
        StrengthTest(network, networkType, checkpoint);
        return true;
    }
    return false;
}

void SelfPlayWorker::StrengthTest(INetwork* network, NetworkType networkType, int step)
{
    std::map<std::string, int> testResults;
    std::map<std::string, int> testPositions;

    std::cout << "Running strength tests..." << std::endl;

    // STS gets special treatment.
    const std::string stsName = "STS";

    // Find strength test .epd files.
    const std::filesystem::path testPath = (Platform::InstallationDataPath() / "StrengthTests");
    for (const auto& entry : std::filesystem::directory_iterator(testPath))
    {
        if (entry.path().extension().string() == ".epd")
        {
            // Hard-coding move times in an ugly way here. They should really be 10-15 seconds for ERET and Arasan20,
            // not 1 second, but this can show some level of progress during training without taking forever.
            // However, only STS results will be comparable to other tested engines.
            const std::string testName = entry.path().stem().string();
            const int moveTimeMs = ((testName == stsName) ? 200 : 1000);

            std::cout << "Testing " << entry.path().filename() << "..." << std::endl;
            const auto [score, total, positions] = StrengthTestEpd(network, networkType, entry.path(), moveTimeMs);
            testResults[testName] = score;
            testPositions[testName] = positions;
        }
    }

    // Estimate an Elo rating using logic here: https://github.com/fsmosca/STS-Rating/blob/master/sts_rating.py
    const float slope = 445.23f;
    const float intercept = -242.85f;
    const float stsRating = (slope * testResults[stsName] / testPositions[stsName]) + intercept;

    // Log to TensorBoard.
    std::vector<std::string> names;
    std::vector<float> values;
    for (const auto& [testName, score] : testResults)
    {
        names.emplace_back("strength/" + testName + "_score");
        values.push_back(static_cast<float>(score));
        std::cout << names.back() << ": " << values.back() << std::endl;
    }
    names.emplace_back("strength/" + stsName + "_rating");
    values.push_back(stsRating);
    std::cout << names.back() << ": " << values.back() << std::endl;
    network->LogScalars(networkType, step, static_cast<int>(names.size()), names.data(), values.data());
}

// Returns (score, total, positions).
std::tuple<int, int, int> SelfPlayWorker::StrengthTestEpd(INetwork* network, NetworkType networkType, const std::filesystem::path& epdPath, int moveTimeMs)
{
    int score = 0;
    int total = 0;
    int positions = 0;

    // Make sure that the prediction cache is clear, for consistent results.
    PredictionCache::Instance.Clear();

    // Warm up the GIL and predictions.
    WarmUpPredictions(network, networkType, 1);

    const std::vector<StrengthTestSpec> specs = Epd::ParseEpds(epdPath);
    positions = static_cast<int>(specs.size());

    for (const StrengthTestSpec& spec : specs)
    {
        const int points = StrengthTestPosition(network, networkType, spec, moveTimeMs);
        score += points;
        total += (spec.points.empty() ? 1 : *std::max_element(spec.points.begin(), spec.points.end()));
    }

    // Clean up after ourselves, e.g. for self-play during training rotations.
    PredictionCache::Instance.Clear();

    return std::tuple(score, total, positions);
}

// For best-move tests returns 1 if correct or 0 if incorrect.
// For points/alternative tests returns N points or 0 if incorrect.
int SelfPlayWorker::StrengthTestPosition(INetwork* network, NetworkType networkType, const StrengthTestSpec& spec, int moveTimeMs)
{
    // Set up the position.
    _games[0].PruneAll();
    SetUpGame(0, spec.fen, {}, true /* tryHard */);

    // Set up search and time control.
    TimeControl timeControl = {};
    timeControl.moveTimeMs = moveTimeMs;

    _searchState.searching = true;
    _searchState.searchStart = std::chrono::high_resolution_clock::now();
    _searchState.lastPrincipleVariationPrint = _searchState.searchStart;
    _searchState.timeControl = timeControl;
    _searchState.nodeCount = 0;
    _searchState.failedNodeCount = 0;
    _searchState.principleVariationChanged = false;

    // Initialize the search.
    const int mctsParallelism = std::min(static_cast<int>(_games.size()), Config::Misc.Search_MctsParallelism);
    SearchInitialize(mctsParallelism);

    // Run the search.
    while (_searchState.searching)
    {
        // Use the specified network type for predictions.
        SearchPlay(mctsParallelism);
        network->PredictBatch(networkType, mctsParallelism, _images.data(), _values.data(), _policies.data());

        // TODO: Only check every N times
        CheckTimeControl();
    }

    // Pick a best move and judge points.
    const Node* bestMove = SelectMove(_games[0]);
    const int points = JudgeStrengthTestPosition(spec, bestMove->move);

    // Free nodes after strength testing (especially for the final position, for which there's no following PruneAll/SetUpGame).
    _games[0].PruneAll();

    return points;
}

int SelfPlayWorker::JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move)
{
    assert(spec.pointSans.empty() ^ spec.avoidSans.empty());
    assert(spec.pointSans.size() == spec.points.size());

    for (const std::string& avoidSan : spec.avoidSans)
    {
        const Move avoid = _games[0].ParseSan(avoidSan);
        assert(avoid != MOVE_NONE);
        if (avoid == move)
        {
            return 0;
        }
    }

    for (int i = 0; i < spec.pointSans.size(); i++)
    {
        const Move bestOrAlternative = _games[0].ParseSan(spec.pointSans[i]);
        assert(bestOrAlternative != MOVE_NONE);
        if (bestOrAlternative == move)
        {
            return spec.points[i];
        }
    }

    if (spec.pointSans.empty() && !spec.avoidSans.empty())
    {
        return 1;
    }
    return 0;
}

void SelfPlayWorker::Play(int index)
{
    SelfPlayState& state = _states[index];
    SelfPlayGame& game = _games[index];

    while (!IsTerminal(game))
    {
        Node* root = game.Root();
        Node* selected = RunMcts(game, _scratchGames[index], _states[index], _mctsSimulations[index], _searchPaths[index], _cacheStores[index]);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }

        assert(selected != nullptr);
        game.StoreSearchStatistics();
        game.ApplyMoveWithRootAndHistory(selected->move, selected);
        game.PruneExcept(root, selected /* == game.Root() */);
        _searchState.principleVariationChanged = true; // First move in PV is now gone.
    }

    // Clean up resources in use and save the result.
    game.Complete();

    state = SelfPlayState::Finished;
}

bool SelfPlayWorker::IsTerminal(const SelfPlayGame& game) const
{
    return (game.Root()->terminalValue.IsImmediate() || (game.Ply() >= Config().SelfPlay.MaxMoves));
}

void SelfPlayWorker::SaveToStorageAndLog(INetwork* network, int index)
{
    const SelfPlayGame& game = _games[index];

    const int ply = game.Ply();
    const float result = game.Result();
    const int gameNumber = _storage->AddTrainingGame(network, game.Save());

    const float gameTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - _gameStarts[index]).count();
    const float mctsTime = (gameTime / ply);
    std::cout << "Game " << gameNumber << ", ply " << ply << ", time " << gameTime << ", mcts time " << mctsTime << ", result " << result << std::endl;
    //PredictionCache::Instance.PrintDebugInfo();
}

void SelfPlayWorker::PredictBatchUniform(int batchSize, INetwork::InputPlanes* /*images*/, float* values, INetwork::OutputPlanes* policies)
{
    std::fill(values, values + batchSize, CHESSCOACH_VALUE_DRAW);

    const int policyCount = (batchSize * INetwork::OutputPlanesFloatCount);
    INetwork::PlanesPointerFlat policiesFlat = reinterpret_cast<INetwork::PlanesPointerFlat>(policies);
    std::fill(policiesFlat, policiesFlat + policyCount, 0.f);
}

Node* SelfPlayWorker::RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation,
    std::vector<Node*>& searchPath, PredictionCacheChunk*& cacheStore)
{
    // Don't get stuck in here forever during search (TryHard) looping on cache hits or terminal nodes.
    // We need to break out and check for PV changes, search stopping, etc. However, need to keep number
    // high enough to get good speed-up from prediction cache hits. Go with 1000 for now.
    const int numSimulations = (game.TryHard() ? (mctsSimulation + 1000) : Config().SelfPlay.NumSimulations);
    for (; mctsSimulation < numSimulations; mctsSimulation++)
    {
        if (state == SelfPlayState::Working)
        {
            // MCTS tree parallelism - enabled when searching, not when training - needs some guidance
            // to avoid repeating the same deterministic child selections:
            // - Avoid branches + leaves by incrementing "visitingCount" while selecting a search path,
            //   lowering the exploration incentive in the UCB score.
            // - However, let searches override this when it's important enough; e.g. going down the
            //   same deep line to explore sibling leaves, or revisiting a checkmate.

            // If parallel MCTS is already expanding the root then we have to just give up this round.
            if (game.Root()->expanding)
            {
                assert(game.TryHard());
                _searchState.failedNodeCount++;
                return nullptr;
            }

            scratchGame = game;
            searchPath.clear();
            searchPath.push_back(scratchGame.Root());
            scratchGame.Root()->visitingCount++;

            while (scratchGame.Root()->IsExpanded())
            {
                // If we can't select a child it's because parallel MCTS is already expanding all
                // children. Give up on this one until next iteration, just fix up visitingCounts.
                Node* selected = SelectChild(scratchGame.Root());
                if (!selected)
                {
                    assert(game.TryHard());
                    for (Node* node : searchPath)
                    {
                        node->visitingCount--;
                    }
                    _searchState.failedNodeCount++;
                    return nullptr;
                }

                scratchGame.ApplyMoveWithRoot(selected->move, selected);
                searchPath.push_back(selected /* == scratchGame.Root() */);
                selected->visitingCount++;
            }
        }

        const bool wasImmediateMate = (scratchGame.Root()->terminalValue == TerminalValue::MateIn<1>());
        float value = scratchGame.ExpandAndEvaluate(state, cacheStore);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            // This is now a dangerous time when searching because this leaf is going to be expanded
            // once the network evaluation/priors come back, but is not yet seen as expanded by
            // parallel searches. Set "expanding" to mark it off-limits.
            scratchGame.Root()->expanding = true;
            return nullptr;
        }

        // Finished actually expanding children, or never needed to wait for an evaluation/priors
        // (e.g. prediction cache hit) or no children possible (terminal node).
        scratchGame.Root()->expanding = false;

        // The value we get is from the final node of the scratch game (could be WHITE or BLACK),
        // from its parent's perspective, and we start applying it at the current position of
        // the actual game (could again be WHITE or BLACK), again from its parent's perspective,
        // so flip it if they differ (the ^). This seems a little strange for the root node, because
        // it doesn't really have a parent in the game, but you can still consider the flipped value
        // as the side-to-play's broad evaluation of the position.
        assert(!std::isnan(value));
        value = SelfPlayGame::FlipValue(Color(game.ToPlay() ^ scratchGame.ToPlay()), value);
        Backpropagate(searchPath, value);
        _searchState.nodeCount++;

        // If we *just found out* that this leaf is a checkmate, prove it backwards as far as possible.
        if (!wasImmediateMate && scratchGame.Root()->terminalValue.IsMateInN())
        {
            BackpropagateMate(searchPath);
        }

        // Adjust best-child pointers (principle variation) now that visits and mates have propagated.
        UpdatePrincipleVariation(searchPath);
        ValidatePrincipleVariation(scratchGame.Root());

        // Expanding the search root is a special case. It happens at the very start of a game,
        // and then whenever a previously-unexplored node is reached as a root (like a 2-repetition,
        // or a child that wasn't visited, but alternatives were getting mated).
        // We need to fix the root's visitCount so that it equals the sum of its children,
        // and correspondingly fix valueSum so that MCTS value (root->Value()) makes sense.
        if (game.Root() == scratchGame.Root())
        {
            assert(game.Root()->visitCount == 1);
            assert(searchPath.size() == 1);
            game.Root()->visitCount = 0;
            game.Root()->valueSum = 0.f;

            // Add exploration noise if not searching.
            if (!game.TryHard())
            {
                AddExplorationNoise(game);
            }
        }
    }

    mctsSimulation = 0;
    return SelectMove(game);
}

void SelfPlayWorker::AddExplorationNoise(SelfPlayGame& game) const
{
    std::gamma_distribution<float> gamma(Config().SelfPlay.RootDirichletAlpha, 1.f);
    std::vector<float> noise(game.Root()->CountChildren());

    float noiseSum = 0.f;
    for (int i = 0; i < noise.size(); i++)
    {
        noise[i] = gamma(Random::Engine);
        noiseSum += noise[i];
    }

    int childIndex = 0;
    for (Node& child : *game.Root())
    {
        const float normalized = (noise[childIndex++] / noiseSum);
        assert(!std::isnan(normalized));
        assert(!std::isinf(normalized));
        child.prior = (child.prior * (1 - Config().SelfPlay.RootExplorationFraction) + normalized * Config().SelfPlay.RootExplorationFraction);
    }
}

Node* SelfPlayWorker::SelectMove(const SelfPlayGame& game) const
{
    if (!game.TryHard() && (game.Ply() < Config().SelfPlay.NumSampingMoves))
    {
        // Use temperature=1; i.e., no need to exponentiate, just use visit counts as the distribution.
        const int sumChildVisits = game.Root()->visitCount;
        int sample = std::uniform_int_distribution<>(0, sumChildVisits - 1)(Random::Engine);
        for (Node& child : *game.Root())
        {
            if (sample < child.visitCount)
            {
                return &child;
            }
            sample -= child.visitCount;
        }
        assert(false);
        return nullptr;
    }
    else
    {
        // Use temperature=0; i.e., just select the best (most-visited, overridden by mates).
        assert(game.Root()->bestChild);
        return game.Root()->bestChild;
    }
}

// It's possible because of nodes marked off-limits via "expanding"
// that this method cannot select a child, instead returning NONE/nullptr.
Node* SelfPlayWorker::SelectChild(Node* parent) const
{
    float maxUcbScore = -std::numeric_limits<float>::infinity();
    Node* max = nullptr;
    for (Node& child : *parent)
    {
        if (!child.expanding)
        {
            const float ucbScore = CalculateUcbScore(parent, &child);
            if (ucbScore > maxUcbScore)
            {
                maxUcbScore = ucbScore;
                max = &child;
            }
        }
    }
    return max;
}

// TODO: Profile, see if significant, whether vectorizing is viable/worth it
float SelfPlayWorker::CalculateUcbScore(const Node* parent, const Node* child) const
{
    // Calculate the exploration rate, which is multiplied by (a) the prior to incentivize exploration,
    // and (b) a mate-in-N lookup to incentivize sufficient exploitation of forced mates, dependent on depth.
    // Include "visitingCount" to help parallel searches diverge.
    const float parentVirtualExploration = static_cast<float>(parent->visitCount + parent->visitingCount);
    const float childVirtualExploration = static_cast<float>(child->visitCount + child->visitingCount);
    const float explorationRate =
        (::logf((parentVirtualExploration + _explorationRateBase + 1.f) / _explorationRateBase) + _explorationRateInit) *
        ::sqrtf(parentVirtualExploration) / (childVirtualExploration + 1.f);

    // (a) prior score
    const float priorScore = explorationRate * child->prior;

    // (b) mate-in-N score
    const float mateScore = child->terminalValue.MateScore(explorationRate);

    return (child->Value() + priorScore + mateScore);
}

void SelfPlayWorker::Backpropagate(const std::vector<Node*>& searchPath, float value)
{
    // Each ply has a different player, so flip each time.
    for (Node* node : searchPath)
    {
        node->visitingCount--;
        node->visitCount++;
        node->valueSum += value;
        value = SelfPlayGame::FlipValue(value);
    }
}

void SelfPlayWorker::BackpropagateMate(const std::vector<Node*>& searchPath)
{
    // To calculate mate values for the tree from scratch we'd need to follow two rules:
    // - If *any* children are a MateIn<N...M> then the parent is an OpponentMateIn<N> (prefer to mate faster).
    // - If *all* children are an OpponentMateIn<N...M> then the parent is a MateIn<M+1> (prefer to get mated slower).
    //
    // However, knowing that values were already correct before, we can just do odd/even checks and stop when nothing changes.
    bool childIsMate = true;
    for (int i = static_cast<int>(searchPath.size()) - 2; i >= 0; i--)
    {
        Node* parent = searchPath[i];

        if (childIsMate)
        {
            // The child in the searchPath just became a mate, or a faster mate.
            // Does this make the parent an opponent mate or faster opponent mate?
            const Node* child = searchPath[i + 1];
            const int newMateN = child->terminalValue.MateN();
            assert(newMateN > 0);
            if (!parent->terminalValue.IsOpponentMateInN() ||
                (newMateN < parent->terminalValue.OpponentMateN()))
            {
                parent->terminalValue = TerminalValue::OpponentMateIn(newMateN);

                // The parent just became worse, so the grandparent may need a different best-child.
                // The regular principle variation update isn't sufficient because it assumes that
                // the search path can only become better than it was.
                const int grandparentIndex = (i - 1);
                if (grandparentIndex >= 0)
                {
                    // It's tempting to try validate the principle variation after this fix, but we
                    // may still be waiting to update it after backpropagating visit counts and mates.
                    // This is only a local fix that ensures that the overall update will be valid.
                    FixPrincipleVariation(searchPath, searchPath[grandparentIndex]);
                }
            }
            else
            {
                return;
            }
        }
        else
        {
            // The child in the searchPath just became an opponent mate or faster opponent mate.
            // Always check all children. This could do nothing, make the parent a new mate, or
            // make the parent a faster mate, depending on which child just got updated.
            int longestChildOpponentMateN = std::numeric_limits<int>::min();
            for (const Node& child : *parent)
            {
                const int childOpponentMateN = child.terminalValue.OpponentMateN();
                if (childOpponentMateN <= 0)
                {
                    return;
                }

                longestChildOpponentMateN = std::max(longestChildOpponentMateN, childOpponentMateN);
            }

            assert(longestChildOpponentMateN > 0);
            parent->terminalValue = TerminalValue::MateIn(longestChildOpponentMateN + 1);
        }

        childIsMate = !childIsMate;
    }
}

void SelfPlayWorker::FixPrincipleVariation(const std::vector<Node*>& searchPath, Node* parent)
{
    bool updatedBestChild = false;
    for (Node& child : *parent)
    {
        if (WorseThan(parent->bestChild, &child))
        {
            parent->bestChild = &child;
            updatedBestChild = true;
        }
    }

    // We updated a best-child, but that only changed the principle variation if this parent was part of it.
    if (updatedBestChild)
    {
        for (int i = 0; i < searchPath.size() - 1; i++)
        {
            if (searchPath[i] == parent)
            {
                _searchState.principleVariationChanged = true;
                break;
            }
            if (searchPath[i]->bestChild != searchPath[i + 1])
            {
                break;
            }
        }
    }
}

void SelfPlayWorker::UpdatePrincipleVariation(const std::vector<Node*>& searchPath)
{
    bool isPrincipleVariation = true;
    for (int i = 0; i < searchPath.size() - 1; i++)
    {
        if (WorseThan(searchPath[i]->bestChild, searchPath[i + 1]))
        {
            searchPath[i]->bestChild = searchPath[i + 1];
            _searchState.principleVariationChanged |= isPrincipleVariation;
        }
        else
        {
            isPrincipleVariation &= (searchPath[i]->bestChild == searchPath[i + 1]);
        }
    }
}

void SelfPlayWorker::ValidatePrincipleVariation(const Node* root)
{
    while (root)
    {
        for (const Node& child : *root)
        {
            if (child.visitCount > 0)
            {
                assert(!WorseThan(root->bestChild, &child));
            }
        }
        root = root->bestChild;
    }
}

bool SelfPlayWorker::WorseThan(const Node* lhs, const Node* rhs) const
{
    // Expect RHS to be defined, so if no LHS then it's better.
    assert(rhs);
    if (!lhs)
    {
        return true;
    }

    // Prefer faster mates and slower opponent mates.
    int lhsEitherMateN = lhs->terminalValue.EitherMateN();
    int rhsEitherMateN = rhs->terminalValue.EitherMateN();
    if (lhsEitherMateN != rhsEitherMateN)
    {
        // For categories (>0, 0, <0), bigger is better.
        // Within categories (1 vs. 3, -2 vs. -4), smaller is better.
        // Add a large term opposing the category sign, then say smaller is better overall.
        lhsEitherMateN += ((lhsEitherMateN < 0) - (lhsEitherMateN > 0)) * 2 * Config().SelfPlay.MaxMoves;
        rhsEitherMateN += ((rhsEitherMateN < 0) - (rhsEitherMateN > 0)) * 2 * Config().SelfPlay.MaxMoves;
        return (lhsEitherMateN > rhsEitherMateN);
    }

    // Prefer more visits.
    return (lhs->visitCount < rhs->visitCount);
}

void SelfPlayWorker::DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut)
{
    if (gameOut) *gameOut = &_games[index];
    if (stateOut) *stateOut = &_states[index];
    if (valuesOut) *valuesOut = &_values[index];
    if (policiesOut) *policiesOut = &_policies[index];
}

SearchState& SelfPlayWorker::DebugSearchState()
{
    return _searchState;
}

// Doesn't try to clear or set up games appropriately, just resets allocations.
void SelfPlayWorker::DebugResetGame(int index)
{
    _games[index] = SelfPlayGame();
    _scratchGames[index] = SelfPlayGame();
}

void SelfPlayWorker::Search(std::function<INetwork*()> networkFactory)
{
    // Create the network on the worker thread (slow).
    std::unique_ptr<INetwork> network(networkFactory());

    // Use the faster student network for UCI predictions.
    const NetworkType networkType = NetworkType_Student;

    // Warm up the GIL and predictions.
    WarmUpPredictions(network.get(), networkType, 1);

    // Start with the position "updated" to the starting position in case of a naked "go" command.
    {
        std::unique_lock lock(_searchConfig.mutexUci);

        if (!_searchConfig.positionUpdated)
        {
            _searchConfig.positionUpdated = true;
            _searchConfig.positionFen = Config::StartingPosition;
            _searchConfig.positionMoves = {};
        }
    }

    // Determine config.
    const int mctsParallelism = std::min(static_cast<int>(_games.size()), Config::Misc.Search_MctsParallelism);

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

            // Wait until told to search or comment.
            while (!_searchConfig.quit && !_searchConfig.search && !_searchConfig.comment)
            {
                _searchConfig.signalUci.wait(lock);
            }
        }

        // Set the position up in _games[0] ready to search or comment.
        UpdatePosition();

        // Commenting is one-shot, jump back above to wait.
        if (_searchConfig.comment.exchange(false))
        {
            CommentOnPosition(network.get());
            continue;
        }

        // Search until stopped.
        UpdateSearch();
        if (_searchState.searching)
        {
            // Initialize the search.
            SearchInitialize(mctsParallelism);

            // Run the search.
            while (!_searchConfig.quit && !_searchConfig.positionUpdated && _searchState.searching)
            {
                SearchPlay(mctsParallelism);
                network->PredictBatch(networkType, mctsParallelism, _images.data(), _values.data(), _policies.data());

                CheckPrintInfo();

                // TODO: Only check every N times
                CheckTimeControl();

                UpdateSearch();
            }
            // Don't free nodes here: leave them available for reuse/freeing with the next UpdatePosition(),
            // and instead clean up below, when quitting.
            OnSearchFinished();
        }
    }

    // Clean up.
    _games[0].PruneAll();
}

// Predicting a batch will trigger the following:
// - initializing Python thread state
// - creating models and loading weights on this thread's assigned TPU/GPU device
// - tracing tf.functions on this thread's assigned TPU/GPU device
PredictionStatus SelfPlayWorker::WarmUpPredictions(INetwork* network, NetworkType networkType, int batchSize)
{
    return network->PredictBatch(networkType, batchSize, _images.data(), _values.data(), _policies.data());
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

        // If the new position is the previous position plus some number of moves,
        // just play out the moves rather than throwing away search results.
        if (_games[0].TryHard() &&
            (_searchState.positionFen == _searchConfig.positionFen) &&
            (_searchConfig.positionMoves.size() >= _searchState.positionMoves.size()) &&
            (std::equal(_searchState.positionMoves.begin(), _searchState.positionMoves.end(), _searchConfig.positionMoves.begin())))
        {
            if (_searchConfig.debug)
            {
                std::cout << "info string [position] Reusing existing position with "
                    << (_searchConfig.positionMoves.size() - _searchState.positionMoves.size()) << " additional moves" << std::endl;
            }
            SetUpGameExisting(0, _searchConfig.positionMoves, static_cast<int>(_searchState.positionMoves.size()));
        }
        else
        {
            if (_searchConfig.debug)
            {
                std::cout << "info string [position] Creating new position" << std::endl;
            }
            _games[0].PruneAll();
            SetUpGame(0, _searchConfig.positionFen, _searchConfig.positionMoves, true /* tryHard */);
        }

        _searchState.positionFen = std::move(_searchConfig.positionFen);
        _searchConfig.positionFen = "";

        _searchState.positionMoves = std::move(_searchConfig.positionMoves);
        _searchConfig.positionMoves = {};

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
            _searchState.lastPrincipleVariationPrint = _searchState.searchStart;
            _searchState.timeControl = _searchConfig.searchTimeControl;
            _searchState.nodeCount = 0;
            _searchState.failedNodeCount = 0;
            _searchState.principleVariationChanged = true; // Print out initial PV.
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
    const Node* bestMove = SelectMove(_games[0]);
    PrintPrincipleVariation();
    std::cout << "bestmove " << UCI::move(bestMove->move, false /* chess960 */) << std::endl;

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
    // Print principle variation when it changes, or at least every 5 seconds.
    if (_searchState.principleVariationChanged ||
        (std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - _searchState.lastPrincipleVariationPrint).count() >= 5.f))
    {
        PrintPrincipleVariation();
        _searchState.principleVariationChanged = false;
    }
}

void SelfPlayWorker::CheckTimeControl()
{
    // Always do at least 1-2 simulations so that a "best" move exists.
    if (!_games[0].Root()->bestChild)
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
        (_searchState.timeControl.timeRemainingMs[toPlay] / Config::Misc.TimeControl_FractionOfRemaining)
        + _searchState.timeControl.incrementMs[toPlay]
        - Config::Misc.TimeControl_SafetyBufferMilliseconds;
    if (timeAllowed > 0)
    {
        if (searchTimeMs >= timeAllowed)
        {
            _searchState.searching = false;
        }
        return;
    }

    // No time allowed at all: defy the system and just make a quick training-style move.
    if (_mctsSimulations[0] >= Config().SelfPlay.NumSimulations)
    {
        _searchState.searching = false;
    }
}

void SelfPlayWorker::PrintPrincipleVariation()
{
    Node* node = _games[0].Root();
    std::vector<Move> principleVariation;

    if (!node->bestChild)
    {
        return;
    }

    while (node->bestChild)
    {
        principleVariation.push_back(node->bestChild->move);
        node = node->bestChild;
    }

    auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::duration sinceSearchStart = (now - _searchState.searchStart);
    _searchState.lastPrincipleVariationPrint = now;

    // Value is from the parent's perspective, so that's already correct for the root perspective
    const Node* pvFirst = _games[0].Root()->bestChild;
    const int eitherMateN = pvFirst->terminalValue.EitherMateN();
    const float value = pvFirst->Value();
    const int depth = static_cast<int>(principleVariation.size());
    const int64_t searchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(sinceSearchStart).count();
    const int nodeCount = _searchState.nodeCount;
    const int nodesPerSecond = static_cast<int>(nodeCount / std::chrono::duration<float>(sinceSearchStart).count());
    const int hashfullPermille = PredictionCache::Instance.PermilleFull();

    std::cout << "info depth " << depth;

    if (eitherMateN != 0)
    {
        std::cout << " score mate " << eitherMateN;
    }
    else
    {
        const int score = static_cast<int>(Game::ProbabilityToCentipawns(value));
        std::cout << " score cp " << score;
    }

    std::cout << " nodes " << nodeCount << " nps " << nodesPerSecond << " time " << searchTimeMs
        << " hashfull " << hashfullPermille << " pv";
    for (Move move : principleVariation)
    {
        std::cout << " " << UCI::move(move, false /* chess960 */);
    }
    std::cout << std::endl;

    // Debug: print cache info.
    if (_searchConfig.debug)
    {
        std::cout << "info string [cache] hitrate " << PredictionCache::Instance.PermilleHits() <<
            " evictionrate " << PredictionCache::Instance.PermilleEvictions() << std::endl;
    }
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

void SelfPlayWorker::SignalComment()
{
    std::lock_guard lock(_searchConfig.mutexUci);

    _searchConfig.comment = true;

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

void SelfPlayWorker::SearchInitialize(int mctsParallelism)
{
    ClearGame(0);

    // Set up parallelism. Make N games share a tree but have their own image/value/policy slots.
    for (int i = 1; i < mctsParallelism; i++)
    {
        ClearGame(i);
        _states[i] = _states[0];
        _gameStarts[i] = _gameStarts[0];
        _games[i] = _games[0].SpawnShadow(&_images[i], &_values[i], &_policies[i]);
    }

    PredictionCache::Instance.ResetProbeMetrics();
}

void SelfPlayWorker::SearchPlay(int mctsParallelism)
{
    for (int i = 0; i < mctsParallelism; i++)
    {
        RunMcts(_games[i], _scratchGames[i], _states[i], _mctsSimulations[i], _searchPaths[i], _cacheStores[i]);
    }
}

void SelfPlayWorker::CommentOnPosition(INetwork* network)
{
    _games[0].GenerateImage(_images[0]);
    const std::vector<std::string> comments = network->PredictCommentaryBatch(1, _images.data());
    std::cout << comments[0] << std::endl;
}