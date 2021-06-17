#include "SelfPlay.h"

#include <limits>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>

#include <Stockfish/thread.h>
#include <Stockfish/uci.h>

#include "Config.h"
#include "Pgn.h"
#include "Random.h"
#include "Syzygy.h"

int8_t TerminalValue::Draw()
{
    return 0;
}

// Mate in N fullmoves, not halfmoves/ply.
int8_t TerminalValue::MateIn(int8_t n)
{
    return n;
}

// Opponent mate in N fullmoves, not halfmoves/ply.
int8_t TerminalValue::OpponentMateIn(int8_t n)
{
    return -n;
}

TerminalValue::TerminalValue()
    : _value()
{
}

TerminalValue::TerminalValue(const int8_t value)
{
    operator=(value);
}

TerminalValue& TerminalValue::operator=(const int8_t value)
{
    _value = value;
    return *this;
}

bool TerminalValue::operator==(const int8_t other) const
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

int8_t TerminalValue::MateN() const
{
    return static_cast<int8_t>(_value ? std::max(0, static_cast<int>(*_value)) : 0);
}

int8_t TerminalValue::OpponentMateN() const
{
    return static_cast<int8_t>(_value ? std::max(0, -*_value) : 0);
}

int8_t TerminalValue::EitherMateN() const
{
    return (_value ? *_value : 0);
}

float TerminalValue::MateScore(float explorationRate) const
{
    const int value = EitherMateN();
    if (value > 0)
    {
        // Encourage visits to faster mates over slower mates. They're more promising (a mate-in-3
        // is more likely to become a mate-in-1 than a mate-in-5) and it helps shape a better
        // training target during self-play. In the end, SelectMove will pick the fastest mate
        // regardless of visits though.
        //
        // This term needs to be multiplied by the exploration rate in order to keep up at high visit counts.
        static_assert((1.f / (1 << 2)) == 0.25f);
        static_assert((1.f / (1 << 3)) == 0.125f);
        const float mateTerm = (value == 1) ? 1.f : (1.f / (1 << value));
        return (explorationRate * mateTerm);
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
        // See Node::SetTerminalValue for more TerminalValue special-casing and explanation.
        //
        // Also, no adjustment for draws at the moment.
        return 0.f;
    }
}

// GCC doesn't like "Node() = default", needs a nested brace for terminalValue.
Node::Node()
    : bestChild{}
    , children{}
    , childCount{}
    , prior{}
    , move{}
    , visitingCount{}
    , visitCount{}
    , valueAverage{}
    , valueWeight{}
    , upWeight{}
    , terminalValue{ {} }
    , expansion{}
    , tablebaseRank{}
    , tablebaseScore{}
    , tablebaseBound{}
{
}

Node::Node(const Node& other)
    : bestChild(other.bestChild.load(std::memory_order_relaxed))
    , children(other.children)
    , childCount(other.childCount)
    , prior(other.prior)
    , move(other.move)
    , visitingCount(other.visitingCount.load(std::memory_order_relaxed))
    , visitCount(other.visitCount.load(std::memory_order_relaxed))
    , valueAverage(other.valueAverage.load(std::memory_order_relaxed))
    , valueWeight(other.valueWeight.load(std::memory_order_relaxed))
    , upWeight(other.upWeight.load(std::memory_order_relaxed))
    , terminalValue(other.terminalValue.load(std::memory_order_relaxed))
    , expansion(other.expansion.load(std::memory_order_relaxed))
    , tablebaseRank(other.tablebaseRank.load(std::memory_order_relaxed))
    , tablebaseScore(other.tablebaseScore.load(std::memory_order_relaxed))
    , tablebaseBound(other.tablebaseBound.load(std::memory_order_relaxed))
{
}

Node::iterator Node::begin()
{
    return children;
}

Node::iterator Node::end()
{
    return (children + childCount);
}

Node::const_iterator Node::begin() const
{
    return children;
}

Node::const_iterator Node::end() const
{
    return (children + childCount);
}

Node::const_iterator Node::cbegin() const
{
    return children;
}

Node::const_iterator Node::cend() const
{
    return (children + childCount);
}

bool Node::IsExpanded() const
{
    return (children != nullptr);
}

float Node::Value() const
{
    // Return a win for a proved mate or loss for a proved opponent mate.
    const int eitherMateN = terminalValue.load(std::memory_order_relaxed).EitherMateN();
    const int mateSign = ((eitherMateN > 0) - (eitherMateN < 0));
    if (mateSign != 0)
    {
        return INetwork::MapProbability11To01(static_cast<float>(mateSign));
    }

    // Initialize "valueAverage" to first-play urgency (FPU) and "valueWeight" to zero.
    // The first "SampleValue" completely clobbers the FPU because of the zero "valueWeight".
    return valueAverage.load(std::memory_order_relaxed);
}

float Node::ValueWithVirtualLoss() const
{
    // Return a win for a proved mate or loss for a proved opponent mate.
    const int eitherMateN = terminalValue.load(std::memory_order_relaxed).EitherMateN();
    const int mateSign = ((eitherMateN > 0) - (eitherMateN < 0));
    if (mateSign != 0)
    {
        return INetwork::MapProbability11To01(static_cast<float>(mateSign));
    }

    // Initialize "valueAverage" to first-play urgency (FPU) and "valueWeight" to zero.
    // The first "SampleValue" completely clobbers the FPU because of the zero "valueWeight".
    const float virtualLossCount = (visitingCount.load(std::memory_order_relaxed) * Config::Network.SelfPlay.VirtualLossCoefficient);
    const float weight = static_cast<float>(valueWeight.load(std::memory_order_relaxed));
    const float safeWeight = std::max(1.f, weight); // Solves non-zero FPU and virtual loss denominator concerns.
    return valueAverage.load(std::memory_order_relaxed) * safeWeight / (safeWeight + virtualLossCount);
}

int Node::SampleValue(float movingAverageBuild, float movingAverageCap, float value)
{
    const int newWeight = (valueWeight.fetch_add(1, std::memory_order_relaxed) + 1);
    float current = valueAverage.load(std::memory_order_relaxed);
    while (!valueAverage.compare_exchange_weak(
        current,
        (current + (value - current) / std::clamp(newWeight * movingAverageBuild, 1.f, movingAverageCap)),
        std::memory_order_relaxed));
    return newWeight;
}

// If needed, this bounding can be merged with mate-proving and TerminalValue, but it all becomes very complicated.
float Node::TablebaseBoundedValue(float value) const
{
    // It would be most correct to write score then bound with release-store, and read score then bound
    // with acquire-load. However, we only backpropagate value, not bounds, so the worst-case impact is
    // low on platforms that don't auto-release. Just take the risk and skip the fence for performance.
    const float score = tablebaseScore.load(std::memory_order_relaxed);
    const Bound bound = tablebaseBound.load(std::memory_order_relaxed);
    switch (bound)
    {
    case BOUND_NONE:
        return value;
    case BOUND_UPPER:
        return std::min(score, value);
    case BOUND_LOWER:
        return std::max(score, value);
    case BOUND_EXACT:
        return score;
    default:
        throw std::runtime_error("Unexpected Bound type");
    }
}

void Node::SetTerminalValue(TerminalValue value)
{
    const TerminalValue previous = terminalValue.exchange(value, std::memory_order_relaxed);
    
    // Clear out almost all prior-based incentive to explore forced losses.
    // This helps avoid wasted visits to high-prior but lost positions, sometimes causing blunders.
    //
    // The search will scramble to visit every other non-losing sibling before returning to known losses,
    // which helps find some alternative value faster, and helps prove forced parent wins faster, but may
    // make it harder to differentiate between faster and slower forced losses (opponent mates). This feels
    // like a very acceptable trade-off.
    //
    // When scrambling, search threads may visit every other sibling and put them into an Expanding
    // state waiting for a neural network evaluation, before returning to the only alternative, the
    // forced loss. This is still safe though because the "blocked" logic in "PuctContext::SelectChild"
    // will kick in, preventing backpropagation, so visits/weight may look high, but "upWeight" will stay
    // low, and value should remain good up the tree. The combination of this and now-very low incentive
    // to visit means that the equivalent of draw-sibling-FPU isn't needed for forced losses.
    //
    // Unfortunately, forced losses will still receive linear exploration incentive, but much later, by design,
    // and this will stop eventually after elimination. Not ideal, but probably not a practical issue,
    // and a little too expensive at runtime to do much about.
    //
    // In order to leave some texture behind, divide the prior by a large number rather than zeroing it.
    //
    // See "TerminalValue::MateScore" for more TerminalValue special-casing and explanation.
    if (!previous.IsOpponentMateInN() && value.IsOpponentMateInN())
    {
        // Only this thread has permission to clobber the non-atomic prior field.
        prior /= 1000.f;
    }
}

void Node::SetTablebaseScoreBound(float score, Bound bound)
{
    // It would be most correct to write score then bound with release-store, and read score then bound
    // with acquire-load. However, we only backpropagate value, not bounds, so the worst-case impact is
    // low on platforms that don't auto-release. Just take the risk and skip the fence for performance.
    tablebaseScore.store(score, std::memory_order_relaxed);
    const Bound previousBound = tablebaseBound.exchange(bound, std::memory_order_relaxed);

    // Follow the same logic as "SetTerminalValue()" for clearing exploration incentive for known losses.
    if ((previousBound != BOUND_UPPER) && (bound == BOUND_UPPER))
    {
        // Only this thread has permission to clobber the non-atomic prior field.
        prior /= 1000.f;
    }
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

// Fast default-constructor with no resource ownership, used to size out vectors.
SelfPlayGame::SelfPlayGame()
    : _root(nullptr)
    , _tryHard(false)
    , _image(nullptr)
    , _value(nullptr)
    , _policy(nullptr)
    , _tablebaseCardinality(nullptr)
    , _searchRootPly(Ply())
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy, int* tablebaseCardinality)
    : Game()
    , _root(new Node())
    , _tryHard(false)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _tablebaseCardinality(tablebaseCardinality)
    , _searchRootPly(Ply())
    , _result(CHESSCOACH_VALUE_UNINITIALIZED)
{
}

SelfPlayGame::SelfPlayGame(const std::string& fen, const std::vector<Move>& moves, bool tryHard,
    INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy, int* tablebaseCardinality)
    : Game(fen, moves)
    , _root(new Node())
    , _tryHard(tryHard)
    , _image(image)
    , _value(value)
    , _policy(policy)
    , _tablebaseCardinality(tablebaseCardinality)
    , _searchRootPly(Ply()) // Important for this to be FEN ply + moves.size() when searching positions.
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
    , _tablebaseCardinality(other._tablebaseCardinality)
    , _searchRootPly(other.Ply()) // Scratch games during MCTS need to snap off higher search roots.
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
    _tablebaseCardinality = other._tablebaseCardinality;
    _searchRootPly = other.Ply(); // Scratch games during MCTS need to snap off higher search roots.
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
    , _tablebaseCardinality(other._tablebaseCardinality)
    , _searchRootPly(other._searchRootPly)
    , _mctsValues(std::move(other._mctsValues))
    , _childVisits(std::move(other._childVisits))
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
    _tablebaseCardinality = other._tablebaseCardinality;
    _searchRootPly = other._searchRootPly;
    _mctsValues = std::move(other._mctsValues);
    _childVisits = std::move(other._childVisits);
    _result = other._result;

    other._root = nullptr;

    return *this;
}

SelfPlayGame::~SelfPlayGame()
{
}

SelfPlayGame SelfPlayGame::SpawnShadow(INetwork::InputPlanes* image, float* value, INetwork::OutputPlanes* policy) const
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

    // Don't prepare an expanded root here because this is a common path; e.g. for scratch games also.
}

void SelfPlayGame::ApplyMoveWithRootAndExpansion(Move move, Node* newRoot, SelfPlayWorker& selfPlayWorker)
{
    ApplyMoveWithRoot(move, newRoot);

    // If this new root is already expanded then we won't hit the "isSearchRoot" expansion in "RunMcts", so perform necessary preparations.
    assert(!TryHard());
    if (newRoot->IsExpanded())
    {
        selfPlayWorker.PrepareExpandedRoot(*this);
    }
}

bool SelfPlayGame::TakeExpansionOwnership(Node* node)
{
    // We already did a relaxed load of "expansion" for this node in the most recent "SelectChild",
    // so move straight to an optimistic "compare_exchange_strong".
    Expansion expected = Expansion::None;
    return node->expansion.compare_exchange_strong(expected, Expansion::Expanding, std::memory_order_relaxed);
}

float SelfPlayGame::ExpandAndEvaluate(SelfPlayState& state, PredictionCacheChunk*& cacheStore, SearchState* searchState,
    bool isSearchRoot, bool generateUniformPredictions)
{
    Node* root = _root;

    // A known-terminal leaf will remain a leaf, so be prepared to
    // quickly return its terminal value on repeated visits.
    const TerminalValue terminalValue = root->terminalValue.load(std::memory_order_relaxed);
    if (terminalValue.IsImmediate())
    {
        state = SelfPlayState::Working;
        return terminalValue.ImmediateValue();
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
    // - network value prediction (valueAverage, back-propagated)
    // - terminal valuation (valueAverage, back-propagated)

    if (state == SelfPlayState::Working)
    {
        // Generate legal moves.
        _expandAndEvaluate_endMoves = generate<LEGAL>(_position, _expandAndEvaluate_moves);

        // Check for checkmate and stalemate.
        const int workingMoveCount = static_cast<int>(_expandAndEvaluate_endMoves - _expandAndEvaluate_moves);
        if (workingMoveCount == 0)
        {
            // Value from the parent's perspective.
            const TerminalValue newTerminalValue = (_position.checkers() ? TerminalValue::MateIn<1>() : TerminalValue::Draw());
            root->SetTerminalValue(newTerminalValue);
            assert(state == SelfPlayState::Working);
            return newTerminalValue.ImmediateValue();
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
            const TerminalValue newTerminalValue = TerminalValue::Draw();
            root->SetTerminalValue(newTerminalValue);
            assert(state == SelfPlayState::Working);
            return newTerminalValue.ImmediateValue();
        }
        const int plyToSearchRoot = (Ply() - _searchRootPly);
        if (IsDrawByTwofoldRepetition(plyToSearchRoot))
        {
            // Value from the parent's perspective (easy, it's a draw).
            // Don't cache the 2-repetition; check again next time.
            assert(root->terminalValue.load(std::memory_order_relaxed).IsNonTerminal());
            assert(state == SelfPlayState::Working);
            return TerminalValue(TerminalValue::Draw()).ImmediateValue();
        }

        // The node isn't terminal, so it's time to take expansion ownership. This protects against cases like:
        // (a) another thread is racing us to expand and will definitely allocate a new Node[] for children
        // (b) another thread just expanded and already allocated a new Node[] for children
        //
        // After successfully taking ownership, either (i) we get a cache hit and immediately expand,
        // or (ii) we set state to WaitingForPrediction, which can imply expansion ownership in future.
        //
        // When failing to take ownership, the state remains Working to prepare for a fresh search path next time.
        if (!TakeExpansionOwnership(root))
        {
            assert(state == SelfPlayState::Working);
            return std::numeric_limits<float>::quiet_NaN();
        }

        // Try get a cached prediction. Only hit the cache up to a max ply for self-play since we
        // see enough unique positions/paths to fill the cache no matter what, and it saves on time
        // to evict less. However, in search (TryHard) it's better to keep everything recent.
        cacheStore = nullptr;
        float cachedValue = std::numeric_limits<float>::quiet_NaN();
        bool hitCached = false;
        if (!generateUniformPredictions &&
            (workingMoveCount <= PredictionCacheEntry::MaxMoveCount) &&
            (TryHard() || (Ply() <= Config::Misc.PredictionCache_MaxPly)))
        {
            // Note that "_imageKey" may be stale whenever "cacheStore" is null.
            _imageKey = GenerateImageKey(TryHard());
            hitCached = PredictionCache::Instance.TryGetPrediction(_imageKey, workingMoveCount,
                &cacheStore, &cachedValue, _cachedPriors.data());
        }
        if (hitCached)
        {
            return FinishExpanding(state, cacheStore, searchState, isSearchRoot, workingMoveCount, cachedValue);
        }

        // Prepare for a prediction from the network.
        //
        // If we're generating uniform predictions, there's no GPU work to do, so just continue onward.
        // (the "prediction" doesn't vary and was already generated when starting this round of self-play).
        // This has the side-effect of each thread just looping over just one game, rather than "prediction_batch_size",
        // which should be more efficient and avoid skewing towards shorter game lengths when stopping early.
        state = SelfPlayState::WaitingForPrediction;
        if (!generateUniformPredictions)
        {
            GenerateImage(*_image);
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    // Received a prediction from the network. WaitingForPrediction implies that we have expansion ownership.
    assert(state == SelfPlayState::WaitingForPrediction);

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

    return FinishExpanding(state, cacheStore, searchState, isSearchRoot, moveCount, value);
}

float SelfPlayGame::FinishExpanding(SelfPlayState& state, PredictionCacheChunk*& cacheStore, SearchState* searchState, bool isSearchRoot, int moveCount, float value)
{
    // Store evaluated value/priors in the cache if appropriate, before any filtering.
    if (cacheStore)
    {
        assert(moveCount <= PredictionCacheEntry::MaxMoveCount);
        cacheStore->Put(_imageKey, value, moveCount, _cachedPriors.data());
    }

    // Handle the UCI "searchmoves" filter at the search root.
    if (isSearchRoot && !searchState->searchMoves.empty())
    {
        // Perform a simplified parallel-array version of std::remove_if, knowing that values are small and primitive.
        float filteredSum = 0.f;
        int replace = 0;
        for (int i = 0; i < moveCount; i++)
        {
            if (std::find(searchState->searchMoves.begin(), searchState->searchMoves.end(), _expandAndEvaluate_moves[i].move) != searchState->searchMoves.end())
            {
                // This move is allowed, so move it to the start. No need to std::move primitives.
                _expandAndEvaluate_moves[replace] = _expandAndEvaluate_moves[i];
                _cachedPriors[replace] = _cachedPriors[i];
                filteredSum += _cachedPriors[i];
                replace++;
            }
        }

        // We know that "searchState->searchMoves" are legal (see "UCI::to_move"), so expect the exact number.
        // Iterate back over the allowed moves and re-normalize their priors.
        moveCount = replace;
        assert(moveCount == searchState->searchMoves.size());
        for (int i = 0; i < replace; i++)
        {
            _cachedPriors[i] /= filteredSum;
        }
    }

    // Expand child nodes with the evaluated or cached priors.
    const float firstPlayUrgency = (isSearchRoot ? CHESSCOACH_FIRST_PLAY_URGENCY_ROOT : CHESSCOACH_FIRST_PLAY_URGENCY_DEFAULT);
    Expand(moveCount, firstPlayUrgency);

    // Probe endgame tablebases for a WDL score for the parent.
    // No need to update "value" here for a successful probe: handled generally in Backpropagate().
    if (Syzygy::ProbeWdl(*this, isSearchRoot))
    {
        searchState->tablebaseHitCount.fetch_add(1, std::memory_order_relaxed);
    }

    state = SelfPlayState::Working;
    return value;
}

void SelfPlayGame::Expand(int moveCount, float firstPlayUrgency)
{
    Node* root = _root;
    assert(!root->IsExpanded());
    assert(moveCount > 0);

    root->children = new Node[moveCount]{};
    root->childCount = moveCount;
    for (int i = 0; i < moveCount; i++)
    {
        root->children[i].move = static_cast<uint16_t>(_expandAndEvaluate_moves[i].move);
        root->children[i].prior = _cachedPriors[i];
        root->children[i].valueAverage = firstPlayUrgency;
    }
}

void SelfPlayGame::DebugExpandCanonicalOrdering()
{
    _expandAndEvaluate_endMoves = generate<LEGAL>(_position, _expandAndEvaluate_moves);
    std::sort(_expandAndEvaluate_moves, _expandAndEvaluate_endMoves, [&](const ExtMove a, const ExtMove b)
        {
            return FlipMove(ToPlay(), a) < FlipMove(ToPlay(), b);
        });
    int moveCount = 0;
    for (ExtMove* cur = _expandAndEvaluate_moves; cur != _expandAndEvaluate_endMoves; cur++)
    {
        _cachedPriors[moveCount] = PolicyValue(*_policy, cur->move); // Logits
        moveCount++;
    }
    Softmax(moveCount, _cachedPriors.data()); // Logits -> priors
    Expand(moveCount, CHESSCOACH_FIRST_PLAY_URGENCY_DEFAULT);
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
        expSum += std::exp(distribution[i] - max);
    }

    const float logSumExp = std::log(expSum) + max;
    for (int i = 0; i < moveCount; i++)
    {
        distribution[i] = std::exp(distribution[i] - logSumExp);
    }
}

float SelfPlayGame::CalculateMctsValue() const
{
    // MCTS is always from the side to play's perspective, matching the root's children's value perspective.
    // Using the best child is the most accurate indicator of the value of the following position/game.
    // Root value underestimates true value because it includes value from poor moves that we don't have to make.
    // Other children may have higher value than the best child, but we're much more sure about the best child's value,
    // whereas others are less explored and have higher value error.
    Node* bestChild = _root->bestChild.load(std::memory_order_relaxed);
    assert(bestChild);
    return bestChild->Value();
}

void SelfPlayGame::StoreSearchStatistics()
{
    std::map<Move, float>& visits = _childVisits.emplace_back();
    float sumChildVisits = 0.f;
    for (const Node& child : *_root)
    {
        sumChildVisits += static_cast<float>(child.visitCount.load(std::memory_order_relaxed));
    }
    for (const Node& child : *_root)
    {
        visits[Move(child.move)] = static_cast<float>(child.visitCount.load(std::memory_order_relaxed)) / sumChildVisits;
    }
    _mctsValues.push_back(CalculateMctsValue());
}

void SelfPlayGame::Complete()
{
    // Save state that depends on nodes.
    // Terminal value is from the parent's perspective, so unconditionally flip (~)
    // from *parent* to *self* before flipping from ToPlay() to white's perspective.
    _result = FlipValue(~ToPlay(), _root->terminalValue.load(std::memory_order_relaxed).ImmediateValue());

    // Clear and detach from all nodes.
    PruneAll();
}

SavedGame SelfPlayGame::Save() const
{
    return SavedGame(Result(), _moves, _mctsValues, _childVisits);
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

    // Hoist "except" from an array member to its own root allocation.
    _root = new Node(*except);

    // Don't let "except"'s descendants get pruned when the original is deleted.
    except->children = nullptr;
    except->childCount = 0;

    // Prune, then update the caller's "except" pointer (now deleted) to the clone.
    PruneAllInternal(root);
    delete root;
    root = nullptr;
    except = _root;
}

void SelfPlayGame::PruneAll()
{
    if (!_root)
    {
        return;
    }

    PruneAllInternal(_root);
    delete _root;
    _root = nullptr;
}

void SelfPlayGame::PruneAllInternal(Node* node)
{
    for (Node& child : *node)
    {
        if (child.IsExpanded())
        {
            PruneAllInternal(&child);
        }
    }
    delete[] node->children;
    node->children = nullptr;
    node->childCount = 0;
}

void SelfPlayGame::AddExplorationNoise()
{
    std::gamma_distribution<float> gamma(Config::Network.SelfPlay.RootDirichletAlpha, 1.f);

    // Use "_cachedPriors" as scratch space.

    float noiseSum = 0.f;
    for (int i = 0; i < _root->childCount; i++)
    {
        _cachedPriors[i] = gamma(Random::Engine);
        noiseSum += _cachedPriors[i];
    }

    int childIndex = 0;
    for (Node& child : *_root)
    {
        const float normalized = (_cachedPriors[childIndex++] / noiseSum);
        assert(!std::isnan(normalized));
        assert(!std::isinf(normalized));
        child.prior = (child.prior * (1 - Config::Network.SelfPlay.RootExplorationFraction) + normalized * Config::Network.SelfPlay.RootExplorationFraction);
    }
}

void SelfPlayGame::UpdateSearchRootPly()
{
    _searchRootPly = Ply();
}

// Only probe for search/UCI, not during self-play.
bool SelfPlayGame::ShouldProbeTablebases()
{
    return TryHard();
}

int& SelfPlayGame::TablebaseCardinality()
{
    return *_tablebaseCardinality;
}

Move SelfPlayGame::ParseSan(const std::string& san)
{
    return Pgn::ParseSan(_position, san);
}

// Don't clear the prediction cache more than once every 5 minutes
// (aimed at preventing N self-play worker threads from each clearing).
Throttle SelfPlayWorker::PredictionCacheResetThrottle(300 * 1000 /* durationMilliseconds */);

void SearchState::Reset(const TimeControl& setTimeControl)
{
    searchStart = std::chrono::high_resolution_clock::now();
    lastPrincipleVariationPrint = searchStart;
    lastBestMove = MOVE_NONE;
    lastBestNodes = 0;
    timeControl = setTimeControl;
    previousNodeCount = 0;
    guiLine.clear();
    guiLineMoves.clear();

    nodeCount = 0;
    failedNodeCount = 0;
    tablebaseHitCount = 0;
    principleVariationChanged = false;
}

SelfPlayWorker::SelfPlayWorker(Storage* storage, SearchState* searchState, int gameCount)
    : _storage(storage)
    , _generateUniformPredictions(false)
    , _states(gameCount)
    , _images(gameCount)
    , _values(gameCount)
    , _policies(gameCount)
    , _tablebaseCardinalities(gameCount)
    , _games(0) // Allocate pooled StateInfos on the worker thread.
    , _scratchGames(0) // Allocate pooled StateInfos on the worker thread.
    , _gameStarts(gameCount)
    , _mctsSimulations(gameCount, 0)
    , _mctsSimulationLimits(gameCount, 0)
    , _searchPaths(gameCount)
    , _cacheStores(gameCount)
    , _searchState(searchState)
    , _currentParallelism(0)
{
}

void SelfPlayWorker::Initialize()
{
    // Allocate pooled StateInfos on the worker thread.
    assert(_games.empty());
    assert(_scratchGames.empty());
    _games.resize(_states.size());
    _scratchGames.resize(_states.size());
}

void SelfPlayWorker::Finalize()
{
    // Deallocate pooled StateInfos on the allocating worker thread.
    _scratchGames.clear();
    _games.clear();
}

void SelfPlayWorker::LoopSelfPlay(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int /* threadIndex */)
{
    Initialize();

    // Wait until games are required.
    while (workCoordinator->WaitForWorkItems())
    {
        // Generate uniform predictions for the first network (rather than use random weights).
        _generateUniformPredictions = workCoordinator->GenerateUniformPredictions();
        if (_generateUniformPredictions)
        {
            // Uniform predictions only need to be set up in memory once while active.
            std::cout << "Generating uniform network predictions until trained" << std::endl;
            PredictBatchUniform(static_cast<int>(_images.size()), _images.data(), _values.data(), _policies.data());
        }

        // Warm up the GIL and predictions.
        if (!_generateUniformPredictions)
        {
            const PredictionStatus warmupStatus = WarmUpPredictions(network, networkType, 1);
            if ((warmupStatus & PredictionStatus_UpdatedNetwork) && PredictionCacheResetThrottle.TryFire())
            {
                // This thread has permission to clear the prediction cache after seeing an updated network.
                std::cout << "Clearing the prediction cache" << std::endl;
                PredictionCache::Instance.Clear();
            }
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
        while (!workCoordinator->AllWorkItemsCompleted())
        {
            // CPU work
            for (int i = 0; i < _games.size(); i++)
            {
                Play(i);

                // In degenerate conditions whole games can finish in CPU via the prediction cache, so loop.
                while ((_states[i] == SelfPlayState::Finished) && !workCoordinator->AllWorkItemsCompleted())
                {
                    SaveToStorageAndLog(network, i);

                    workCoordinator->OnWorkItemCompleted();

                    SetUpGame(i);
                    Play(i);
                }
            }

            // GPU work
            if (!_generateUniformPredictions)
            {
                const PredictionStatus status = network->PredictBatch(networkType, static_cast<int>(_images.size()), _images.data(), _values.data(), _policies.data());
                if ((status & PredictionStatus_UpdatedNetwork) && PredictionCacheResetThrottle.TryFire())
                {
                    // This thread has permission to clear the prediction cache after seeing an updated network.
                    std::cout << "Clearing the prediction cache" << std::endl;
                    PredictionCache::Instance.Clear();
                }
            }
        }

        // Don't free nodes here. Let the games and MCTS trees be continued using the next network
        // after clearing the prediction cache (via PredictionStatus flag).
    }

    Finalize();
}

int SelfPlayWorker::ChooseSimulationLimit()
{
    return Config::Network.SelfPlay.NumSimulations;
}

void SelfPlayWorker::ClearGame(int index)
{
    _states[index] = SelfPlayState::Working;
    _gameStarts[index] = std::chrono::high_resolution_clock::now();
    _mctsSimulations[index] = 0;
    _mctsSimulationLimits[index] = ChooseSimulationLimit();
    _searchPaths[index].clear();
    _cacheStores[index] = nullptr;
}

void SelfPlayWorker::SetUpGame(int index)
{
    ClearGame(index);
    _games[index] = SelfPlayGame(&_images[index], &_values[index], &_policies[index], &_tablebaseCardinalities[index]);
}

void SelfPlayWorker::SetUpGame(int index, const std::string& fen, const std::vector<Move>& moves, bool tryHard)
{
    ClearGame(index);
    _games[index] = SelfPlayGame(fen, moves, tryHard, &_images[index], &_values[index], &_policies[index], &_tablebaseCardinalities[index]);
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
            newRoot = ((i == (moves.size() - 1)) ? new Node() : nullptr);
            game.PruneAll();
            game.ApplyMoveWithRoot(move, newRoot);
        }
    }

    // The additional moves are being applied to an existing game for efficiency, but really we're setting up
    // a new position, so make necessary updates.
    UpdateGameForNewSearchRoot(game);
}

void SelfPlayWorker::TrainNetwork(INetwork* network, NetworkType networkType, int step, int checkpoint)
{
    // Delegate to Python.
    std::cout << "Training steps " << step << "-" << checkpoint << "..." << std::endl;
    auto startTrain = std::chrono::high_resolution_clock::now();
    network->Train(networkType, step, checkpoint);
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

void SelfPlayWorker::SaveSwaNetwork(INetwork* network, NetworkType networkType, int checkpoint)
{
    // Save the network to (a) allow stopping and resuming, and (b) give the prediction network the latest weights.
    network->SaveSwaNetwork(networkType, checkpoint);
}

void SelfPlayWorker::StrengthTestNetwork(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int checkpoint)
{
    std::cout << "Running strength tests..." << std::endl;

    // Only run STS in the interest of time.
    const std::string stsName = "STS";
    const int moveTimeMs = 200;
    const std::filesystem::path epdPath = (Platform::InstallationDataPath() / "StrengthTests" / "STS.epd");
    std::cout << "Testing " << epdPath.filename() << "..." << std::endl;
    const auto [score, total, positions, totalNodesRequired] = StrengthTestEpd(workCoordinator, epdPath,
        moveTimeMs, 0 /* nodes */, 0 /* failureNodes */, 0 /* positionLimit */, nullptr /* progress */);

    // Estimate an Elo rating using logic here: https://github.com/fsmosca/STS-Rating/blob/master/sts_rating.py
    const float slope = 445.23f;
    const float intercept = -242.85f;
    const float stsRating = (slope * score / positions) + intercept;

    // Log to TensorBoard.
    std::vector<std::string> names;
    std::vector<float> values;
    names.emplace_back("strength/" + stsName + "_score");
    names.emplace_back("strength/" + stsName + "_rating");
    values.push_back(static_cast<float>(score));
    values.push_back(stsRating);
    for (int i = 0; i < names.size(); i++)
    {
        std::cout << names[i] << ": " << values[i] << std::endl;
    }
    network->LogScalars(networkType, checkpoint, names, values.data());
}

// Returns (score, total, positions, totalNodesRequired).
std::tuple<int, int, int, int> SelfPlayWorker::StrengthTestEpd(WorkCoordinator* workCoordinator, const std::filesystem::path& epdPath,
    int moveTimeMs, int nodes, int failureNodes, int positionLimit,
    std::function<void(const std::string&, const std::string&, const std::string&, int, int, int)> progress)
{
    int score = 0;
    int total = 0;
    int positions = 0;
    int totalNodesRequired = 0;

    // Make sure that the prediction cache is clear, for consistent results.
    // It would be better to clear before every position in the EPD, but too slow.
    PredictionCache::Instance.Clear();

    const std::vector<StrengthTestSpec> specs = Epd::ParseEpds(epdPath);
    positions = static_cast<int>(specs.size());
    if (positionLimit > 0)
    {
        positions = std::min(positions, positionLimit);
    }

    for (int i = 0; i < positions; i++)
    {
        const StrengthTestSpec& spec = specs[i];
        const auto [move, points, nodesRequired] = StrengthTestPosition(workCoordinator, spec, moveTimeMs, nodes, failureNodes);
        const int available = (spec.points.empty() ? 1 : *std::max_element(spec.points.begin(), spec.points.end()));
        score += points;
        total += available;
        totalNodesRequired += nodesRequired;

        if (progress)
        {
            // This target string is a bit of a minimal hack, not as general as scoring behavior.
            const std::string target = !spec.pointSans.empty() ? spec.pointSans.front() : ("avoid " + spec.avoidSans.front());
            const std::string chosen = Pgn::San(spec.fen, move, false /* showCheckmate */); // Assume no mate-in-ones.
            progress(spec.fen, target, chosen, points, available, nodesRequired);
        }
    }

    // Clean up after ourselves, e.g. for self-play during training rotations.
    PredictionCache::Instance.Clear();

    if (failureNodes == 0)
    {
        totalNodesRequired = 0;
    }

    return { score, total, positions, totalNodesRequired };
}

// For best-move tests returns 1 if correct or 0 if incorrect.
// For points/alternative tests returns N points or 0 if incorrect.
std::tuple<Move, int, int> SelfPlayWorker::StrengthTestPosition(WorkCoordinator* workCoordinator, const StrengthTestSpec& spec, int moveTimeMs, int nodes, int failureNodes)
{
    // Make sure that the workers are ready.
    workCoordinator->WaitForWorkers();

    // Set up the position.
    SearchUpdatePosition(std::string(spec.fen), {}, true /* forceNewPosition */);

    // Set up search and time control.
    TimeControl timeControl = {};
    timeControl.moveTimeMs = moveTimeMs;
    timeControl.nodes = nodes;
    _searchState->Reset(timeControl);

    // Run the search.
    workCoordinator->ResetWorkItemsRemaining(1);
    workCoordinator->WaitForWorkers();

    // Pick a best move and judge points.
    const Node* best = SelectMove(_games[0], false /* allowDiversity */);
    const Move bestMove = Move(best->move);
    const auto [points, nodesRequired] = JudgeStrengthTestPosition(spec, bestMove, _searchState->lastBestNodes, failureNodes);

    // Free nodes after strength testing (especially for the final position, for which there's no following PruneAll/SetUpGame).
    _games[0].PruneAll();

    return { bestMove, points, nodesRequired };
}

std::pair<int, int> SelfPlayWorker::JudgeStrengthTestPosition(const StrengthTestSpec& spec, Move move, int lastBestNodes, int failureNodes)
{
    assert(spec.pointSans.empty() ^ spec.avoidSans.empty());
    assert(spec.pointSans.size() == spec.points.size());

    for (const std::string& avoidSan : spec.avoidSans)
    {
        const Move avoid = _games[0].ParseSan(avoidSan);
        assert(avoid != MOVE_NONE);
        if (avoid == move)
        {
            return { 0, failureNodes };
        }
    }

    const auto bestPoints = std::max_element(spec.points.begin(), spec.points.end());
    for (int i = 0; i < spec.pointSans.size(); i++)
    {
        const Move bestOrAlternative = _games[0].ParseSan(spec.pointSans[i]);
        assert(bestOrAlternative != MOVE_NONE);
        if (bestOrAlternative == move)
        {
            return { spec.points[i], (spec.points[i] == *bestPoints) ? lastBestNodes : failureNodes };
        }
    }

    if (spec.pointSans.empty() && !spec.avoidSans.empty())
    {
        // There's no granular node information for avoid-move (am) positions.
        return { 1, 0 };
    }
    return { 0, failureNodes };
}

void SelfPlayWorker::Play(int index)
{
    SelfPlayState& state = _states[index];
    SelfPlayGame& game = _games[index];

    while (!IsTerminal(game))
    {
        Node* root = game.Root();
        const bool mctsFinished = RunMcts(game, _scratchGames[index], _states[index], _mctsSimulations[index],
            _mctsSimulationLimits[index], _searchPaths[index], _cacheStores[index], false /* finishOnly */);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            return;
        }

        // During self-play there are no threading-based node failures, so if we're not waiting for a prediction,
        // assert that MCTS finished successfully and we can pick a move.
        Node* selected = SelectMove(game, true /* allowDiversity */);
        (void)mctsFinished;
        assert(mctsFinished);
        assert(selected != nullptr);
        game.StoreSearchStatistics();
        game.ApplyMoveWithRootAndExpansion(Move(selected->move), selected, *this);
        game.PruneExcept(root, selected /* == game.Root() */);
        // Use release-store to synchronize with the acquire-load of the PV printing so that the PV is updated.
        _searchState->principleVariationChanged.store(true, std::memory_order_release); // First move in PV is now gone.
    }

    // Clean up resources in use and save the result.
    game.Complete();

    state = SelfPlayState::Finished;
}

bool SelfPlayWorker::IsTerminal(const SelfPlayGame& game) const
{
    return (game.Root()->terminalValue.load(std::memory_order_relaxed).IsImmediate() || (game.Ply() >= Config::Network.SelfPlay.MaxMoves));
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
}

void SelfPlayWorker::PredictBatchUniform(int batchSize, INetwork::InputPlanes* /*images*/, float* values, INetwork::OutputPlanes* policies)
{
    std::fill(values, values + batchSize, CHESSCOACH_VALUE_DRAW);

    const int policyCount = (batchSize * INetwork::OutputPlanesFloatCount);
    INetwork::PlanesPointerFlat policiesFlat = reinterpret_cast<INetwork::PlanesPointerFlat>(policies);
    std::fill(policiesFlat, policiesFlat + policyCount, 0.f);
}

bool SelfPlayWorker::RunMcts(SelfPlayGame& game, SelfPlayGame& scratchGame, SelfPlayState& state, int& mctsSimulation, int& mctsSimulationLimit,
    std::vector<WeightedNode>& searchPath, PredictionCacheChunk*& cacheStore, bool finishOnly)
{
    // Don't get stuck in here forever during search (TryHard) looping on cache hits or terminal nodes.
    // We need to break out and check for PV changes, search stopping, etc. However, need to keep number
    // high enough to get good speed-up from prediction cache hits. Go with 1000 for now.
    if (game.TryHard())
    {
        mctsSimulation = 0;
        mctsSimulationLimit = 1000;
    }
    for (; mctsSimulation < mctsSimulationLimit; mctsSimulation++)
    {
        if (state == SelfPlayState::Working)
        {
            // When parallel games are used to search a position during UCI or strength testing, it's best to
            // redeem knowledge from nodes that were waiting on a network prediction before selecting new ones.
            if (finishOnly)
            {
                return false;
            }

            // MCTS tree parallelism - enabled when searching, not when training - needs some guidance
            // to avoid repeating the same deterministic child selections:
            // - Avoid branches + leaves by incrementing "visitingCount" while selecting a search path,
            //   lowering the exploration incentive and value calculation in the PUCT score.
            // - However, let searches override this when it's important enough; e.g. going down the
            //   same deep line to explore sibling leaves, or revisiting a checkmate.

            scratchGame = game;
            assert(searchPath.empty());
            searchPath.clear();
            searchPath.push_back({ scratchGame.Root(), 1 });
            scratchGame.Root()->visitingCount.fetch_add(1, std::memory_order_relaxed);

            // We need this acquire-load to synchronize with the release-store of the expanding thread
            // so that the side-effects - children - are visible here.
            while (scratchGame.Root()->expansion.load(std::memory_order_acquire) == Expansion::Expanded)
            {
                // If we can't select a child it's because parallel MCTS is already expanding all
                // children. Give up on this one until next iteration.
                WeightedNode selected = PuctContext(_searchState, scratchGame.Root()).SelectChild();
                if (!selected.node)
                {
                    assert(game.TryHard());
                    FailNode(searchPath);
                    return false;
                }

                scratchGame.ApplyMoveWithRoot(Move(selected.node->move), selected.node);
                searchPath.push_back(selected /* == scratchGame.Root() */);
                selected.node->visitingCount.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Call in to ExpandAndEvaluate straight away, since we want to allow multiple threads/games in to visit terminal nodes
        // like wins simultaneously, and there's no other clean/efficient way to deal with situations like transient 2-repetition draws
        // without thrashing the "expansion" flag.
        //
        // However, before allocating and expanding into "children", take ownership of the expansion inside ExpandAndEvaluate,
        // and give up if beaten to it by another thread.
        //
        // Despite these complications, it's best to keep terminal and non-terminal evaluation consolidated within ExpandAndEvaluate
        // because beyond trivially cached terminal evaluations, both depend on move generation.
        const bool wasImmediateMate = (scratchGame.Root()->terminalValue.load(std::memory_order_relaxed) == TerminalValue::MateIn<1>());
        const bool isSearchRoot = (game.Root() == scratchGame.Root());
        float value = scratchGame.ExpandAndEvaluate(state, cacheStore, _searchState, isSearchRoot, _generateUniformPredictions);
        if (state == SelfPlayState::WaitingForPrediction)
        {
            // Wait for network evaluation/priors to come back.
            return false;
        }
        else if (std::isnan(value))
        {
            // Another thread took ownership and is expanding or expanded the node. We have to just give up this round.
            assert(game.TryHard());
            FailNode(searchPath);
            return false;
        }

        // The value we get is from the final node of the scratch game from its parent's perspective,
        // so start applying it there and work backwards. This seems a little strange for the root node, because
        // it doesn't really have a parent in the game, but you can still consider the flipped value
        // as the side-to-play's broad evaluation of the position.
        assert(!std::isnan(value));
        // Also get the root's value from the leaf's perspective in case it's needed for draw-sibling-FPU (see inside Backpropagate()).
        const float rootValue = Game::FlipValue(Color(game.ToPlay() ^ scratchGame.ToPlay()), game.Root()->valueAverage.load(std::memory_order_relaxed));
        assert(!std::isnan(rootValue));
        Backpropagate(searchPath, value, rootValue);
        _searchState->nodeCount.fetch_add(1, std::memory_order_relaxed);

        // If we *just found out* that this leaf is a checkmate, prove it backwards as far as possible.
        if (!wasImmediateMate && scratchGame.Root()->terminalValue.load(std::memory_order_relaxed).IsMateInN())
        {
            BackpropagateMate(searchPath);
        }

        // Adjust best-child pointers (principle variation) now that visits and mates have propagated.
        UpdatePrincipleVariation(searchPath);
        ValidatePrincipleVariation(game.Root());

        // Expanding the search root is a special case. It happens at the very start of a game,
        // and then whenever a previously-unexpanded node is reached as a root (like a 2-repetition,
        // or a child that wasn't visited, but alternatives were getting mated). Fix the visitCount to 1
        // so that exploration incentives are normal, in case the node had a lot of 2-repetition terminal visits.
        // Note that we don't generally update previously-2-repetition visits, just when they become search roots.
        if (isSearchRoot)
        {
            // Only make these corrections when freshly expanding, not inside "PrepareExpandedRoot" for already-expanded children,
            // in which case we want to preserve visits, etc., so that exploration continues normally.
            assert(searchPath.size() == 1);
            game.Root()->visitCount.store(1, std::memory_order_relaxed);
            game.Root()->valueAverage.store(CHESSCOACH_FIRST_PLAY_URGENCY_ROOT, std::memory_order_relaxed); // Meaningless, but be consistent.
            game.Root()->valueWeight.store(0, std::memory_order_relaxed);
            game.Root()->upWeight.store(0, std::memory_order_relaxed);

            // Perform necessary preparations, like adding exploration noise for self-play games.
            PrepareExpandedRoot(game);
        }

        // Clear the search path after decrementing "visitingCount" to avoid duplicate decrement on search stop/resume.
        searchPath.clear();

        // If this node has children then we must have just finished expanding it. Advance the expansion flag
        // and release-store to synchronize with the acquire-load of the search-path building and move selection.
        if (scratchGame.Root()->IsExpanded())
        {
            scratchGame.Root()->expansion.store(Expansion::Expanded, std::memory_order_release);
        }
    }

    // Self-play resets the simulation count/limit here between moves within a game.
    mctsSimulation = 0;
    mctsSimulationLimit = ChooseSimulationLimit();
    return true;
}

void SelfPlayWorker::FailNode(std::vector<WeightedNode>& searchPath)
{
    for (auto [node, weight] : searchPath)
    {
        node->visitingCount.fetch_sub(1, std::memory_order_relaxed);
    }
    searchPath.clear();
    _searchState->failedNodeCount.fetch_add(1, std::memory_order_relaxed);
}

void SelfPlayWorker::UpdateGameForNewSearchRoot(SelfPlayGame& game)
{
    // Update the search root ply for draw-checking.
    game.UpdateSearchRootPly();

    // If this new root is already expanded then we won't hit the "isSearchRoot" expansion in "RunMcts", so perform necessary preparations.
    assert(game.TryHard());
    if (game.Root()->IsExpanded())
    {
        PrepareExpandedRoot(game);
    }
}

// This is a common path for self-play and search, (a) when expanding an initial/updated root, or (b) when updating the root to an already-expanded child.
// - For both self-play and search, case (a) is recognized by "isSearchRoot" in "RunMcts".
// - For self-play, case (b) will almost always be hit after the starting position, since each immediate child should be visited at least once, and we reuse the MCTS tree.
// - For search, case (b) will be hit when we're able to reuse an existing position; e.g., after moves have been played, or searching deeper positions without a "ucinewgame".
void SelfPlayWorker::PrepareExpandedRoot(SelfPlayGame& game)
{
    // Set first-play urgency (FPU) to a win here for children of the root.
    for (Node& child : *game.Root())
    {
        // Only one thread can be working on this node: we've either just made a move in self-play/UCI, which is a single-threaded pause
        // between multi-threaded work, or we've taken expansion ownership until we go back and set Expansion::Expanded.
        //
        // Therefore, it's safe to replace whatever FPU was there, as long as "valueWeight" is zero (e.g. replace draw-sibling-FPU, which could be anything).
        //
        // We may overwrite a root-probed tablebase score, but it will be restored on the first backpropagation because of the
        // bounded value (see SelfPlayWorker::Backpropagate) and zero "valueWeight" (see Node::SampleValue).
        if (child.valueWeight.load(std::memory_order_relaxed) == 0)
        {
            child.valueAverage.store(CHESSCOACH_FIRST_PLAY_URGENCY_ROOT, std::memory_order_relaxed);
        }
    }

    // Add exploration noise if not searching.
    if (!game.TryHard())
    {
        game.AddExplorationNoise();
    }

    // Probing endgame tablebases at the root is important for "last-resort" move selection
    // in case the search doesn't find a good line concretely; e.g., especially when the
    // no-progress count won't reset anymore, like in KBN vs. K.
    //
    // For endgames like King/Bishop/Knight vs. King (KBNK) we need distance-to-zero (DTZ) information
    // in order to choose moves that preserve the win, not just based on the piece layout, but piece layout plus
    // current 50-move/no-progress count.
    //
    // When there are too many pieces at the root to probe endgame tablebases, we can still try
    // to probe individual leaf positions when they reach few enough pieces. We only probe win/draw/loss (WDL)
    // "at zero", when progress has just been made (pawn move or capture). Accurate search is still very necessary.
    if (Syzygy::ProbeTablebasesAtRoot(game))
    {
        // In addition to setting tablebase ranks, which we use even before proven mate categories for move selection,
        // we just updated root child value: not just FPU, but bounded value for nodes with existing valueWeight too.
        // Fix up the principle variation to take all of this into account. This may result in a "bestChild", or
        // collected best children, with zero visits, which needs to be handled carefully.
        _searchState->tablebaseHitCount.fetch_add(game.Root()->childCount, std::memory_order_relaxed);
        FixPrincipleVariation({ { game.Root(), 0 }, { game.Root(), 0 } }, game.Root());
    }
}

Node* SelfPlayWorker::SelectMove(const SelfPlayGame& game, bool allowDiversity) const
{
    Node* bestChild = game.Root()->bestChild.load(std::memory_order_relaxed);
    if (!bestChild)
    {
        // Haven't had enough time to explore anything, so just pick the highest prior.
        // We need this acquire-load to synchronize with the release-store of the expanding thread
        // so that the side-effects - children - are visible here.
        if (game.Root()->expansion.load(std::memory_order_acquire) == Expansion::Expanded)
        {
            Node* bestPrior = nullptr;
            for (Node& child : *game.Root())
            {
                if (!bestPrior || (bestPrior->prior < child.prior))
                {
                    bestPrior = &child;
                }
            }
            assert(bestPrior);
            return bestPrior;
        }
        else
        {
            // No legal moves or priors visible, so just return a MOVE_NONE via the root.
            return game.Root();
        }
    }
    else if (!game.TryHard() && allowDiversity && (game.Ply() < Config::Network.SelfPlay.NumSampingMoves))
    {
        // Sample using temperature=1, treating normalized visit counts as a probability distribution
        // (like when they're passed as truth labels to cross-entropy loss). So, no need to exponentiate.
        std::vector<int> weights(game.Root()->childCount);
        for (int i = 0; i < game.Root()->childCount; i++)
        {
            weights[i] = game.Root()->children[i].visitCount.load(std::memory_order_relaxed);
        }
        std::discrete_distribution distribution(weights.begin(), weights.end());
        return &game.Root()->children[distribution(Random::Engine)];
    }
    else if (game.TryHard() && allowDiversity && (game.Ply() < Config::Network.SelfPlay.MoveDiversityPlies) &&
        (Config::Network.SelfPlay.MoveDiversityTemperature > 0.f))
    {
        // "When we forced AlphaZero to play with greater diversity (by softmax sampling with a temperature of 10.0 among moves
        // for which the value was no more than 1% away from the best move for the first 30 plies) the winning rate increased from 5.8% to 14%."
        //
        // We can use a similar system, but SBLE-PUCT introduces some complications:
        // - Visits become very even at high node counts because of linear exploration.
        // - UpWeight should be more representative of "deserved" visits, but isn't because there's no backpropagation catch-up for unlucky ordering.
        // So, sample based on visits, but greatly lower the softmax sampling temperature.
        std::vector<Node*> best = CollectBestMoves(game.Root(), Config::Network.SelfPlay.MoveDiversityValueDeltaThreshold);
        if (best.size() == 1)
        {
            return best.back();
        }
        // CollectBestMoves only selects from the same tablebase rank/mate categories as "bestMove", so "bestMove" has the most visits among the collected.
        // Re-exponentiate using different temperature. |(N/max(N))^(1/t)| gives the same result as softmax(log(N)/t) but skips the logarithm.
        //
        // Set visit counts to at least one in case nodes are best-collected because of a root tablebase probe.
        const float bestVisitCount = static_cast<float>(std::max(1, bestChild->visitCount.load(std::memory_order_relaxed)));
        std::vector<float> bestWeights(best.size());
        for (int i = 0; i < best.size(); i++)
        {
            bestWeights[i] = std::pow(static_cast<float>(std::max(1, best[i]->visitCount.load(std::memory_order_relaxed))) / bestVisitCount, 1.f / Config::Network.SelfPlay.MoveDiversityTemperature);
        }
        std::discrete_distribution distribution(bestWeights.begin(), bestWeights.end());
        return best[distribution(Random::Engine)];
    }
    else
    {
        // Use temperature=0; i.e., just select the best (most-visited, overridden by mates).
        return bestChild;
    }
}

thread_local std::vector<ScoredNode> PuctContext::ScoredNodes;

PuctContext::PuctContext(const SearchState* searchState, Node* parent)
    : _parent(parent)
{
    // Pre-compute repeatedly-used terms.
    _parentVirtualExploration = VirtualExploration(parent);

    const float explorationRateBase = Config::Network.SelfPlay.ExplorationRateBase;
    const float explorationRateInit = Config::Network.SelfPlay.ExplorationRateInit;
    _explorationNumerator =
        (std::log((_parentVirtualExploration + explorationRateBase + 1.f) / explorationRateBase) + explorationRateInit) *
        std::sqrt(_parentVirtualExploration);

    // At the root, use "eliminationFraction" to exponentially decay down from the top 64 to top 2 children based on AZ-PUCT score,
    // with only the top K receiving SBLE-PUCT linear exploration incentive.
    //
    // At other nodes with fewer visits, eliminate less harshly early on, based on the fraction of root visits.
    //
    // Bound to at least 2, then at most the child count.
    const int eliminationExponent = std::max(1, Config::Network.SelfPlay.EliminationBaseExponent -
        static_cast<int>(searchState->timeControl.eliminationFraction * Config::Network.SelfPlay.EliminationBaseExponent));
    const int parentVisitCount = std::max(1, _parent->visitCount.load(std::memory_order_relaxed));
    const int rootVisitCount = std::max(parentVisitCount, searchState->timeControl.eliminationRootVisitCount);
    const int rootVisitAdjusted = std::min((1 << Config::Network.SelfPlay.EliminationBaseExponent),
        ((1 << eliminationExponent) * rootVisitCount / parentVisitCount));
    _eliminationTopCount = std::min(_parent->childCount, rootVisitAdjusted);

    // Localize repeatedly-used config.
    _linearExplorationRate = Config::Network.SelfPlay.LinearExplorationRate;
    _linearExplorationBase = Config::Network.SelfPlay.LinearExplorationBase;
}

// It's possible because of nodes marked off-limits via "expanding"
// that this method cannot select a child, instead returning NONE/nullptr.
WeightedNode PuctContext::SelectChild() const
{
    float maxAzPuct = -std::numeric_limits<float>::infinity();
    float maxSblePuct = -std::numeric_limits<float>::infinity();
    float azOfMaxSble = -std::numeric_limits<float>::infinity();
    float maxSblePuctIncludingBlocked = -std::numeric_limits<float>::infinity();
    Node* maxSble = nullptr;
    bool bestWasBlocked = false;

    // It might be better to keep children sorted/pivoted, but that's not viable with multiple threads mutating scores.
    ScoredNodes.clear();
    for (Node& child : *_parent)
    {
        const float childVirtualExploration = VirtualExploration(&child);
        const float azPuct = CalculateAzPuctScore(&child, childVirtualExploration);
        maxAzPuct = std::max(maxAzPuct, azPuct);
        ScoredNodes.emplace_back(&child, azPuct, childVirtualExploration);
    }
    std::nth_element(ScoredNodes.begin(), ScoredNodes.begin() + _eliminationTopCount, ScoredNodes.end());

    for (int i = 0; i < ScoredNodes.size(); i++)
    {
        Node* child = ScoredNodes[i].node;
        const float azPuct = ScoredNodes[i].score;
        const float sblePuct = (i < _eliminationTopCount) ? CalculateSblePuctScore(azPuct, ScoredNodes[i].virtualExploration) : azPuct;
        if (sblePuct > maxSblePuct)
        {
            // Can also include other gates here, like flood protection in small sub-trees.
            const bool blocked = (child->expansion.load(std::memory_order_relaxed) == Expansion::Expanding);
            if (!blocked)
            {
                maxSblePuct = sblePuct;
                azOfMaxSble = azPuct;
                maxSble = child;
            }
            if (sblePuct > maxSblePuctIncludingBlocked)
            {
                maxSblePuctIncludingBlocked = sblePuct;
                bestWasBlocked = blocked;
            }
        }
    }

    // Select child using max(SBLE-PUCT), but only backpropagate value if its AZ-PUCT is within range of max(AZ-PUCT).
    const int weight = (!bestWasBlocked) & ((azOfMaxSble / maxAzPuct) >= Config::Network.SelfPlay.BackpropagationPuctThreshold);
    return { maxSble, weight };
}

// AZ-PUCT is the AlphaZero Predictor-Upper Confidence bound applied to Trees (with a mate-term modification and virtual exploration/loss).
float PuctContext::CalculateAzPuctScore(const Node* child, float childVirtualExploration) const
{
    // Calculate the exploration rate, which is multiplied by (a) the prior to incentivize exploration,
    // and (b) a mate-in-N lookup to incentivize sufficient exploitation of forced mates, dependent on depth.
    // Include "visitingCount" to help parallel searches diverge ("virtual exploration").
    const float explorationRate = _explorationNumerator / (childVirtualExploration + 1.f);

    // (a) prior score
    const float priorScore = explorationRate * child->prior;

    // (b) mate-in-N score
    const float mateScore = child->terminalValue.load(std::memory_order_relaxed).MateScore(explorationRate);

    return (child->ValueWithVirtualLoss() + priorScore + mateScore);
}

// SBLE-PUCT is the Selective-Backpropagation, Linear Exploration, Predictor-Upper Confidence bound applied to Trees.
float PuctContext::CalculateSblePuctScore(float azPuctScore, float childVirtualExploration) const
{
    // Calculate SBLE-PUCT linear term.
    const float linear = _parentVirtualExploration / ((_linearExplorationRate * childVirtualExploration) + _linearExplorationBase);

    return (azPuctScore + linear);
}

float PuctContext::CalculatePuctScoreAdHoc(const Node* child) const
{
    // AZ-PUCT makes the most sense to display.
    return CalculateAzPuctScore(child, VirtualExploration(child));
}

float PuctContext::VirtualExploration(const Node* node) const
{
    return static_cast<float>(node->visitCount.load(std::memory_order_relaxed) + node->visitingCount.load(std::memory_order_relaxed));
}

void SelfPlayWorker::Backpropagate(std::vector<WeightedNode>& searchPath, float value, float rootValue)
{
    // Each ply has a different player, so flip each time.
    const float movingAverageBuild = Config::Network.SelfPlay.MovingAverageBuild;
    const float movingAverageCap = Config::Network.SelfPlay.MovingAverageCap;
    int weight = 1;
    Node* previous = nullptr;
    for (int i = static_cast<int>(searchPath.size()) - 1; i >= 0; i--)
    {
        if (!weight)
        {
            BackpropagateVisitsOnly(searchPath, i);
            return;
        }

        Node* node = searchPath[i].node;
        node->visitingCount.fetch_sub(1, std::memory_order_relaxed);
        node->visitCount.fetch_add(1, std::memory_order_relaxed);
         
        // If this node has a tablebase score/bound set then we know we can achieve at least/exactly/at most that,
        // so backpropagate the bounded value up the tree.
        //
        // Also, if this node has a tablebase score/bound set and we picked a sub-par exploration-only node down below
        // (i.e. weight is now zero, and we didn't reach this point), we could restart the weight chain, since we know
        // that the bounded value accurately represents the node. However, this has performance impact, and may not be necessary.
        value = node->TablebaseBoundedValue(value);

        const int newWeight = node->SampleValue(movingAverageBuild, movingAverageCap, value);
        if (previous)
        {
            previous->upWeight.fetch_add(1, std::memory_order_relaxed);
        }

        // Implement "draw-sibling-FPU":
        //
        // When a leaf node is first visited and the value is an exact draw, update the first-play urgency (FPU)
        // of the draw's siblings to the root's value (flipped if necessary), so that if a player has a winning position,
        // they are not forced to keep visiting and backpropagating a draw that was encountered by surprise with a high prior.
        // This removes the need for flood protection in small sub-trees, and solves more problems in practice.
        //
        // There are two potential issues with this logic:
        // - The neural network evaluation or quantized prediction cache evaluation may exactly equal a draw score.
        //   This is unfortunate, but only results in a small slow-down for the initial visits to siblings with
        //   sufficient exploration incentive.
        // - If another thread just backpropagated into a sibling with an exact loss score as its first (few) sample(s)
        //   then we may clobber the actual weighted average, because we can only CAS against one variable. This could be
        //   dangerous, delaying future visits to the win potentially for a long time, but should be exceedingly rare.
        if ((newWeight == 1) && (value == CHESSCOACH_VALUE_DRAW) && (i == static_cast<int>(searchPath.size()) - 1) && (i > 0))
        {
            // Iterate over siblings.
            for (Node& child : *searchPath[i - 1].node)
            {
                if (child.valueWeight.load(std::memory_order_relaxed) == 0)
                {
                    float expected = CHESSCOACH_FIRST_PLAY_URGENCY_DEFAULT;
                    child.valueAverage.compare_exchange_strong(expected, rootValue, std::memory_order_relaxed);
                }
            }
        }
        value = Game::FlipValue(value);

        // Weights are always 0 or 1 with current SBLE-PUCT. We already checked the current weight, so no need for "std::min".
        weight = searchPath[i].weight;
        previous = node;
    }
}

void SelfPlayWorker::BackpropagateVisitsOnly(std::vector<WeightedNode>& searchPath, int index)
{
    for (; index >= 0; index--)
    {
        Node* node = searchPath[index].node;
        node->visitingCount.fetch_sub(1, std::memory_order_relaxed);
        node->visitCount.fetch_add(1, std::memory_order_relaxed);
    }
}

void SelfPlayWorker::BackpropagateMate(const std::vector<WeightedNode>& searchPath)
{
    // To calculate mate values for the tree from scratch we'd need to follow two rules:
    // - If *any* children are a MateIn<N...M> then the parent is an OpponentMateIn<N> (prefer to mate faster).
    // - If *all* children are an OpponentMateIn<N...M> then the parent is a MateIn<M+1> (prefer to get mated slower).
    //
    // However, knowing that values were already correct before, we can just do odd/even checks and stop when nothing changes.
    bool childIsMate = true;
    for (int i = static_cast<int>(searchPath.size()) - 2; i >= 0; i--)
    {
        Node* parent = searchPath[i].node;
        const TerminalValue parentTerminalValue = parent->terminalValue.load(std::memory_order_relaxed);

        if (childIsMate)
        {
            // The child in the searchPath just became a mate, or a faster mate.
            // Does this make the parent an opponent mate or faster opponent mate?
            const Node* child = searchPath[i + 1].node;
            const TerminalValue childTerminalValue = child->terminalValue.load(std::memory_order_relaxed);
            const int8_t newMateN = childTerminalValue.MateN();
            assert(newMateN > 0);
            if (!parentTerminalValue.IsOpponentMateInN() ||
                (newMateN < parentTerminalValue.OpponentMateN()))
            {
                parent->SetTerminalValue(TerminalValue::OpponentMateIn(newMateN));

                // The parent just became worse, so the grandparent may need a different best-child.
                // The regular principle variation update isn't sufficient because it assumes that
                // the search path can only become better than it was.
                const int grandparentIndex = (i - 1);
                if (grandparentIndex >= 0)
                {
                    // It's tempting to try validate the principle variation after this fix, but we
                    // may still be waiting to update it after backpropagating visit counts and mates.
                    // This is only a local fix that ensures that the overall update will be valid.
                    FixPrincipleVariation(searchPath, searchPath[grandparentIndex].node);
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
            int8_t longestChildOpponentMateN = std::numeric_limits<int8_t>::min();
            for (const Node& child : *parent)
            {
                const TerminalValue childTerminalValue = child.terminalValue.load(std::memory_order_relaxed);
                const int8_t childOpponentMateN = childTerminalValue.OpponentMateN();
                if (childOpponentMateN <= 0)
                {
                    return;
                }

                longestChildOpponentMateN = std::max(longestChildOpponentMateN, childOpponentMateN);
            }

            assert(longestChildOpponentMateN > 0);
            parent->SetTerminalValue(TerminalValue::MateIn(longestChildOpponentMateN + 1));
        }

        childIsMate = !childIsMate;
    }
}

void SelfPlayWorker::FixPrincipleVariation(const std::vector<WeightedNode>& searchPath, Node* parent)
{
    // We may update the best child multiple times in this loop. E.g. just discovered current best loses to mate,
    // then loop through and see 9 visits, then 10 visits, then 11 visits.
    bool updateBestChild = false;
    Node* parentBestChild = parent->bestChild.load(std::memory_order_relaxed);
    for (Node& child : *parent)
    {
        if (WorseThan(parentBestChild, &child))
        {
            parentBestChild = &child;
            updateBestChild = true;
        }
    }

    // We're updating a best-child, but that only changes the principle variation if this parent was part of it.
    if (updateBestChild)
    {
        parent->bestChild.store(parentBestChild, std::memory_order_relaxed);
        for (int i = 0; i < searchPath.size() - 1; i++)
        {
            if (searchPath[i].node == parent)
            {
                // Use release-store to synchronize with the acquire-load of the PV printing so that the PV is updated.
                _searchState->principleVariationChanged.store(true, std::memory_order_release);
                break;
            }
            if (searchPath[i].node->bestChild.load(std::memory_order_relaxed) != searchPath[i + 1].node)
            {
                break;
            }
        }
    }
}

void SelfPlayWorker::UpdatePrincipleVariation(const std::vector<WeightedNode>& searchPath)
{
    bool isPrincipleVariation = true;
    for (int i = 0; i < searchPath.size() - 1; i++)
    {
        if (WorseThan(searchPath[i].node->bestChild.load(std::memory_order_relaxed), searchPath[i + 1].node))
        {
            searchPath[i].node->bestChild.store(searchPath[i + 1].node, std::memory_order_relaxed);
            if (isPrincipleVariation)
            {
                // Use release-store to synchronize with the acquire-load of the PV printing so that the PV is updated.
                _searchState->principleVariationChanged.store(true, std::memory_order_release);
            }
        }
        else
        {
            isPrincipleVariation &= (searchPath[i].node->bestChild.load(std::memory_order_relaxed) == searchPath[i + 1].node);
        }
    }
}

void SelfPlayWorker::ValidatePrincipleVariation(const Node* root)
{
    // Principle variation may be temporarily invalid from a search thread's perspective when multiple are running.
    if (_games[0].TryHard() && (Config::Misc.Search_SearchThreads > 1))
    {
        return;
    }

    while (root)
    {
        const Node* bestChild = root->bestChild.load(std::memory_order_relaxed);
        for (const Node& child : *root)
        {
            if (child.visitCount.load(std::memory_order_relaxed) > 0)
            {
                assert(!WorseThan(bestChild, &child));
            }
        }
        root = bestChild;
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

    // It's important to differentiate using tablebase rank first, when known, since this gives absolute
    // guarantees on preserving a win or draw when distance-to-zero (DTZ) information is available.
    //
    // When ranks are the same - e.g. two different wins with no repetition since last progress - we can use
    // proven mates discovered through search to choose a faster win.
    //
    // With perfect search, we could find a faster mate than the "safe" Syzygy route, since Syzygy tables don't
    // include distance-to-mate (DTM) information. However, if we considered proven mate information before
    // tablebase rank we could end up drawing via the 50-move rule, so choose the safest route.
    if (lhs->tablebaseRank != rhs->tablebaseRank)
    {
        return lhs->tablebaseRank < rhs->tablebaseRank;
    }

    // Prefer faster mates and slower opponent mates.
    int lhsEitherMateN = lhs->terminalValue.load(std::memory_order_relaxed).EitherMateN();
    int rhsEitherMateN = rhs->terminalValue.load(std::memory_order_relaxed).EitherMateN();
    if (lhsEitherMateN != rhsEitherMateN)
    {
        // For categories (>0, 0, <0), bigger is better.
        // Within categories (1 vs. 3, -2 vs. -4), smaller is better.
        // Add a large term opposing the category sign, then say smaller is better overall.
        lhsEitherMateN += ((lhsEitherMateN < 0) - (lhsEitherMateN > 0)) * 2 * Config::Network.SelfPlay.MaxMoves;
        rhsEitherMateN += ((rhsEitherMateN < 0) - (rhsEitherMateN > 0)) * 2 * Config::Network.SelfPlay.MaxMoves;
        return (lhsEitherMateN > rhsEitherMateN);
    }

    // Prefer more visits.
    return (lhs->visitCount.load(std::memory_order_relaxed) < rhs->visitCount.load(std::memory_order_relaxed));
}

std::vector<Node*> SelfPlayWorker::CollectBestMoves(Node* parent, float valueDeltaThreshold) const
{
    Node* bestChild = parent->bestChild.load(std::memory_order_relaxed);
    assert(bestChild);
    std::vector<Node*> best = { bestChild };
    const int bestTablebaseRank = bestChild->tablebaseRank.load(std::memory_order_relaxed);
    const int bestEitherMateN = bestChild->terminalValue.load(std::memory_order_relaxed).EitherMateN();
    const float valueThreshold = (bestChild->Value() - valueDeltaThreshold);
    for (Node& child : *parent)
    {
        // Other candidates must be in the same tablebase rank and mate categories as the best child.
        if ((&child == bestChild) ||
            (child.tablebaseRank.load(std::memory_order_relaxed) != bestTablebaseRank) ||
            (child.terminalValue.load(std::memory_order_relaxed).EitherMateN() != bestEitherMateN))
        {
            continue;
        }

        // Other candidates must be within an absolute value difference:
        // "By 1% we mean in absolute value. All our values are between 0 and 1, so if the best move has a value of 0.8, we would sample from all moves with values >= 0.79."
        if (child.Value() > valueThreshold)
        {
            best.push_back(&child);
        }
    }
    return best;
}

void SelfPlayWorker::DebugGame(int index, SelfPlayGame** gameOut, SelfPlayState** stateOut, float** valuesOut, INetwork::OutputPlanes** policiesOut)
{
    if (gameOut) *gameOut = &_games[index];
    if (stateOut) *stateOut = &_states[index];
    if (valuesOut) *valuesOut = &_values[index];
    if (policiesOut) *policiesOut = &_policies[index];
}

// Doesn't try to clear or set up games appropriately, just resets allocations.
void SelfPlayWorker::DebugResetGame(int index)
{
    _games[index] = SelfPlayGame();
    _scratchGames[index] = SelfPlayGame();
}

void SelfPlayWorker::LoopSearch(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int threadIndex)
{
    const bool primary = (threadIndex == 0);
    Initialize();

    // Warm up the GIL and predictions.
    WarmUpPredictions(network, networkType, 1);

    // Wait until searching is required.
    while (workCoordinator->WaitForWorkItems())
    {
        // Initialize the search. Multiple threads will race to make shadows of the reference position,
        // which is safe because the shallow fields don't mutate. Care just needs to be taken with the
        // shared Node tree.
        SearchInitialize(_searchState->position);

        // Search until stopped.
        while (!workCoordinator->AllWorkItemsCompleted())
        {
            // CPU work
            if (!SearchPlay(threadIndex))
            {
                continue;
            }

            // GPU work
            network->PredictBatch(networkType, _currentParallelism, _images.data(), _values.data(), _policies.data());

            // Only the primary worker does housekeeping.
            if (primary)
            {
                CheckPrincipleVariation();

                CheckUpdateGui(network, false /* forceUpdate */);

                CheckTimeControl(workCoordinator);
            }
        }

        // Let the original position owner free nodes via SearchUpdatePosition(), but fix up node visits/expansions in flight.
        FinalizeMcts();

        // Only the primary worker does housekeeping.
        if (primary)
        {
            CheckUpdateGui(network, true /* forceUpdate */);
            OnSearchFinished();
        }
    }

    Finalize();
}

void SelfPlayWorker::LoopStrengthTest(WorkCoordinator* workCoordinator, INetwork* network, NetworkType networkType, int threadIndex)
{
    const bool primary = (threadIndex == 0);
    Initialize();

    // Warm up the GIL and predictions.
    WarmUpPredictions(network, networkType, 1);

    // Wait until searching is required.
    while (workCoordinator->WaitForWorkItems())
    {
        // Initialize the search. Multiple threads will race to make shadows of the reference position,
        // which is safe because the shallow fields don't mutate. Care just needs to be taken with the
        // shared Node tree.
        SearchInitialize(_searchState->position);

        // Search until stopped.
        while (!workCoordinator->AllWorkItemsCompleted())
        {
            // CPU work
            if (!SearchPlay(threadIndex))
            {
                continue;
            }

            // GPU work
            network->PredictBatch(networkType, _currentParallelism, _images.data(), _values.data(), _policies.data());

            // Only the primary worker does housekeeping.
            if (primary)
            {
                // Update "lastBestNodes" for strength tests.
                const bool principleVariationChanged = _searchState->principleVariationChanged.exchange(false, std::memory_order_acquire);
                if (principleVariationChanged)
                {
                    const uint16_t newBest = _games[0].Root()->bestChild.load(std::memory_order_relaxed)->move;
                    if (newBest != _searchState->lastBestMove)
                    {
                        _searchState->lastBestMove = newBest;
                        _searchState->lastBestNodes = _searchState->nodeCount;
                    }
                }

                CheckTimeControl(workCoordinator);
            }
        }

        // Let the original position owner free nodes via SearchUpdatePosition(), but fix up node visits/expansions in flight.
        FinalizeMcts();

        // Only the primary worker does housekeeping.
        if (primary)
        {
            // Report the best move to bot code in Python.
            if (!_searchState->botGameId.empty() && !_searchState->timeControl.infinite)
            {
                const Node* bestMove = SelectMove(_games[0], true /* allowDiversity */);
                const std::string& bestMoveUci = UCI::move(Move(bestMove->move), false /* chess960 */);
                network->PlayBotMove(_searchState->botGameId, bestMoveUci);
            }
        }
    }

    Finalize();
}

bool SelfPlayWorker::SearchPlay(int threadIndex)
{
    // Finish off MCTS for any nodes that were waiting on a network prediction by expanding, backpropagating, etc.,
    // across all parallel games. This gives us maximum knowledge for the selection of new nodes.
    for (int i = 0; i < _currentParallelism; i++)
    {
        RunMcts(_games[i], _scratchGames[i], _states[i], _mctsSimulations[i], _mctsSimulationLimits[i], _searchPaths[i], _cacheStores[i], true /* finishOnly */);
    }

    // Get maximum throughput in tiny time controls and avoid misshapen MCTS trees by limiting parallelism
    // early on and not flooding nodes into a small tree ("slowstart" feature). This could also be implemented
    // throughout the tree (see in "PuctContext::SelectChild"), but it doesn't seem to help so far.
    const int nodeCount = _games[0].Root()->visitCount.load(std::memory_order_relaxed); // Requires "RunMcts" with "finishOnly" to have just run.
    int parallelism = static_cast<int>(_games.size());
    if (nodeCount < Config::Misc.Search_SlowstartNodes)
    {
        // This thread may not be needed yet.
        if (threadIndex >= Config::Misc.Search_SlowstartThreads)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // ~1-15 milliseconds
            return false;
        }

        // This thread is needed, but limit parallelism.
        parallelism = std::min(parallelism, Config::Misc.Search_SlowstartParallelism);
    }

    // Now we can select new nodes based on latest knowledge and chosen parallelism. Cache hits and terminals can still be finished and keep looping.
    _currentParallelism = parallelism;
    for (int i = 0; i < parallelism; i++)
    {
        RunMcts(_games[i], _scratchGames[i], _states[i], _mctsSimulations[i], _mctsSimulationLimits[i], _searchPaths[i], _cacheStores[i], false /* finishOnly */);
    }
    
    return true;
}

// Predicting a batch will trigger the following:
// - initializing Python thread state
// - creating models and loading weights on this thread's assigned TPU/GPU device
// - tracing tf.functions on this thread's assigned TPU/GPU device
PredictionStatus SelfPlayWorker::WarmUpPredictions(INetwork* network, NetworkType networkType, int batchSize)
{
    return network->PredictBatch(networkType, batchSize, _images.data(), _values.data(), _policies.data());
}

void SelfPlayWorker::SearchUpdatePosition(const std::string& fen, const std::vector<Move>& moves, bool forceNewPosition)
{
    // If the new position is the previous position plus some number of moves,
    // just play out the moves rather than throwing away search results.
    if (!forceNewPosition &&
        _games[0].TryHard() &&
        (fen == _searchState->positionFen) &&
        (moves.size() >= _searchState->positionMoves.size()) &&
        (std::equal(_searchState->positionMoves.begin(), _searchState->positionMoves.end(), moves.begin())))
    {
        if (_searchState->debug.load(std::memory_order_relaxed))
        {
            std::cout << "info string [position] Reusing existing position with "
                << (moves.size() - _searchState->positionMoves.size()) << " additional moves" << std::endl;
        }
        SetUpGameExisting(0, moves, static_cast<int>(_searchState->positionMoves.size()));
    }
    else
    {
        if (_searchState->debug.load(std::memory_order_relaxed))
        {
            std::cout << "info string [position] Creating new position" << std::endl;
        }
        _games[0].PruneAll();
        SetUpGame(0, fen, moves, true /* tryHard */);
    }

    _searchState->position = &_games[0];
    _searchState->positionFen = fen; // Copy
    _searchState->positionMoves = moves; // Copy
}

void SelfPlayWorker::FinalizeMcts()
{
    // We may reuse this search tree on the next UCI search if the position is compatible, and likely have
    // node visits and expansions in flight that we just interrupted, so fix everything up.
    for (int i = 0; i < _searchPaths.size(); i++)
    {
        for (auto [node, weight] : _searchPaths[i])
        {
            node->visitingCount.fetch_sub(1, std::memory_order_relaxed);
            Expansion expected = Expansion::Expanding;
            node->expansion.compare_exchange_strong(expected, Expansion::None, std::memory_order_relaxed);
        }
        _searchPaths[i].clear();
    }
}

void SelfPlayWorker::OnSearchFinished()
{
    // Print the final PV info and bestmove.
    const Node* bestMove = SelectMove(_games[0], true /* allowDiversity */);
    PrintPrincipleVariation(true /* searchFinished */);
    std::cout << "bestmove " << UCI::move(Move(bestMove->move), false /* chess960 */) << std::endl;
}

void SelfPlayWorker::CheckPrincipleVariation()
{
    // Print principle variation when it changes, or at least every 5 seconds.
    // Use acquire-load to synchronize with the release-store of updaters so that side effects - the PV - are visible.
    const bool principleVariationChanged = _searchState->principleVariationChanged.exchange(false, std::memory_order_acquire);
    if (principleVariationChanged ||
        (std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - _searchState->lastPrincipleVariationPrint).count() >= 5.f))
    {
        PrintPrincipleVariation(false /* searchFinished */);
    }
}

void SelfPlayWorker::CheckUpdateGui(INetwork* network, bool forceUpdate)
{
    const int interval = Config::Misc.Search_GuiUpdateIntervalNodes;
    const int nodeCount = _searchState->nodeCount.load(std::memory_order_relaxed);
    if (_searchState->gui && (forceUpdate ||
        ((nodeCount / interval) > (_searchState->previousNodeCount / interval))))
    {
        // Drill down to the requested line.
        SelfPlayGame lineGame = _games[0];
        for (const Move move : _searchState->guiLineMoves)
        {
            lineGame.ApplyMoveWithRoot(move, lineGame.Root()->Child(move));
        }

        // Wait to update the GUI until a principle variation exists.
        const Node* root = lineGame.Root();
        const Node* bestChild = root->bestChild.load(std::memory_order_relaxed);
        if (!bestChild)
        {
            return;
        }

        const std::string fen = lineGame.GetPosition().fen();

        // Value is from the parent's perspective, so that's already correct for the root perspective
        std::stringstream evaluation;
        const int eitherMateN = bestChild->terminalValue.load(std::memory_order_relaxed).EitherMateN();
        const float pvValue = bestChild->Value();
        if (eitherMateN != 0)
        {
            evaluation << std::fixed << std::setprecision(6) << pvValue
                << " (" << ((eitherMateN > 0) ? "mate in " : "opponent mate in ") << std::abs(eitherMateN) << ")";
        }
        else
        {
            evaluation << std::fixed << std::setprecision(6) << pvValue
                << " (" << (Game::ProbabilityToCentipawns(pvValue) / 100.f) << " pawns)";
        }

        // Compose a SAN principle variation: more expensive but only used for GUI and only every "Search_GuiUpdateIntervalNodes".
        std::stringstream principleVariation;
        SelfPlayGame pvGame = lineGame;
        const Node* pvBestChild = bestChild;
        while (pvBestChild)
        {
            const Move move = Move(pvBestChild->move);
            const std::string san = Pgn::San(pvGame.GetPosition(), move,
                (pvBestChild->terminalValue.load(std::memory_order_relaxed) == TerminalValue::MateIn<1>()) /* showCheckmate */);
            principleVariation << san << " ";
            pvGame.ApplyMove(move);
            pvBestChild = pvBestChild->bestChild.load(std::memory_order_relaxed);
        }

        std::vector<std::string> sans;
        std::vector<std::string> froms;
        std::vector<std::string> tos;
        std::vector<float> targets;
        std::vector<float> priors;
        std::vector<float> values;
        std::vector<float> puct;
        std::vector<int> visits;
        std::vector<int> weights;
        std::vector<int> upWeights;

        float sumChildVisits = 0.f;
        for (const Node& child : *lineGame.Root())
        {
            sumChildVisits += static_cast<float>(child.visitCount.load(std::memory_order_relaxed));
        }
        PuctContext puctContext(_searchState, lineGame.Root());
        for (const Node& child : *lineGame.Root())
        {
            const Move move = Move(child.move);
            sans.emplace_back(Pgn::San(lineGame.GetPosition(), move, (child.terminalValue.load(std::memory_order_relaxed) == TerminalValue::MateIn<1>()) /* showCheckmate */));
            froms.emplace_back(Game::SquareName[from_sq(move)]);
            tos.emplace_back(Game::SquareName[to_sq(move)]);
            targets.push_back(static_cast<float>(child.visitCount.load(std::memory_order_relaxed)) / sumChildVisits);
            priors.push_back(child.prior);
            values.push_back(child.Value());
            puct.push_back(puctContext.CalculatePuctScoreAdHoc(&child));
            visits.push_back(child.visitCount.load(std::memory_order_relaxed));
            weights.push_back(child.valueWeight.load(std::memory_order_relaxed));
            upWeights.push_back(child.upWeight.load(std::memory_order_relaxed));
        }

        network->UpdateGui(fen, _searchState->guiLine, nodeCount, evaluation.str(), principleVariation.str(),
            sans, froms, tos, targets, priors, values, puct, visits, weights, upWeights);
    }
    _searchState->previousNodeCount = nodeCount;
}

void SelfPlayWorker::GuiShowLine(INetwork* network, const std::string& line)
{
    if (line == _searchState->guiLine)
    {
        return;
    }

    // Parse the line SANs.
    std::vector<Move> lineMoves;
    SelfPlayGame lineGame = _games[0];
    std::stringstream lineSans(line);
    std::string san;
    while (lineSans >> san)
    {
        lineMoves.emplace_back(Pgn::ParseSan(lineGame.GetPosition(), san));
        const Move& move = lineMoves.back(); // Reference returned by "emplace_back" may be invalid on MSVC when reallocating.
        lineGame.ApplyMoveWithRoot(move, lineGame.Root()->Child(move));

        // Don't show lines for unexpanded nodes, or nodes with no principle variation.
        const Node* bestChild = lineGame.Root()->bestChild.load(std::memory_order_relaxed);
        if (!lineGame.Root()->IsExpanded() || !bestChild)
        {
            return;
        }
    }

    // Send an update to the GUI using the new "guiLineMoves".
    _searchState->guiLine = line;
    _searchState->guiLineMoves = std::move(lineMoves);
    CheckUpdateGui(network, true /* forceUpdate */);
}

void SelfPlayWorker::CheckTimeControl(WorkCoordinator* workCoordinator)
{
    // Always try to do at least 1-2 simulations so that a "best" move exists.
    // Note that this may not be possible because of a hard "stop" or "position" command,
    // so SelectMove, PrintPrincipleVariation and OnSearchFinished handle the case of no bestChild.
    const Node* root = _games[0].Root();
    const Node* bestChild = root->bestChild.load(std::memory_order_relaxed);
    if (!bestChild)
    {
        // Stop despite any other instructions (e.g. infinite) if the root is terminal.
        if (root->terminalValue.load(std::memory_order_relaxed).IsImmediate())
        {
            workCoordinator->OnWorkItemCompleted();
        }
        return;
    }

    // Infinite think takes priority: nothing else can stop the search.
    if (_searchState->timeControl.infinite)
    {
        return;
    }

    // Mate can stop the search.
    if (_searchState->timeControl.mate > 0)
    {
        const int eitherMateN = bestChild->terminalValue.load(std::memory_order_relaxed).EitherMateN();
        if ((eitherMateN > 0) && (eitherMateN <= _searchState->timeControl.mate))
        {
            workCoordinator->OnWorkItemCompleted();
            return;
        }
    }

    const std::chrono::duration sinceSearchStart = (std::chrono::high_resolution_clock::now() - _searchState->searchStart);
    const int64_t searchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(sinceSearchStart).count();

    // Nodes deeper in the tree with fewer visits receive less harsh elimination. Capture the baseline.
    _searchState->timeControl.eliminationRootVisitCount = root->visitCount.load(std::memory_order_relaxed);

    // Nodes can stop the search.
    const int nodeCount = _searchState->nodeCount.load(std::memory_order_relaxed);
    if (_searchState->timeControl.nodes > 0)
    {
        if (nodeCount >= _searchState->timeControl.nodes)
        {
            workCoordinator->OnWorkItemCompleted();
            return;
        }

        // We are "eliminationFraction" of the way through the search, based on nodes.
        _searchState->timeControl.eliminationFraction = (static_cast<float>(nodeCount) / _searchState->timeControl.nodes);
    }

    // Specified think time can stop the search.
    if (_searchState->timeControl.moveTimeMs > 0)
    {
        const int64_t timeAllowed = _searchState->timeControl.moveTimeMs;
        if (searchTimeMs >= timeAllowed)
        {
            workCoordinator->OnWorkItemCompleted();
            return;
        }

        // We are "eliminationFraction" of the way through the search, based on move time.
        _searchState->timeControl.eliminationFraction = (static_cast<float>(searchTimeMs) / timeAllowed);
    }

    // Game clock can stop the search. Use a simple strategy like AlphaZero for now.
    const Color toPlay = _games[0].ToPlay();
    const int64_t totalTimeAllowed = (_searchState->timeControl.timeRemainingMs[toPlay]);
    if (totalTimeAllowed > 0)
    {
        int fraction = Config::Misc.TimeControl_FractionOfRemaining;
        if (_searchState->timeControl.movesToGo > 0)
        {
            // If it's 40 moves per 5 min with 2 moves/60 seconds remaining, use 30 seconds.
            fraction = std::min(fraction, _searchState->timeControl.movesToGo);
        }

        // Use a fraction of the increment-free remaining time, plus the increment,
        // but definitely use at most the remaining time (if there's a bug).
        const int64_t increment = _searchState->timeControl.incrementMs[toPlay];
        const int64_t excludingIncrement = std::max(0LL, totalTimeAllowed - increment);
        const int64_t fractionPlusIncrement = ((excludingIncrement / fraction) + increment);
        const int64_t timeAllowed =
            (std::min(fractionPlusIncrement, totalTimeAllowed)
            - Config::Misc.TimeControl_SafetyBufferMilliseconds);
        if (searchTimeMs >= timeAllowed)
        {
            workCoordinator->OnWorkItemCompleted();
            return;
        }

        // In game clock mode, time can always be saved up for future moves, so if there's only one legal move then make it.
        if (root->IsExpanded() && (root->childCount == 1))
        {
            workCoordinator->OnWorkItemCompleted();
            return;
        }

        // We are "eliminationFraction" of the way through the search, based on time remaining and simple time control strategy.
        _searchState->timeControl.eliminationFraction = (static_cast<float>(searchTimeMs) / timeAllowed);
    }

    // No limits set/remaining: make at least the training number of simulations.
    if ((_searchState->timeControl.nodes <= 0) &&
        (_searchState->timeControl.mate <= 0) &&
        (_searchState->timeControl.moveTimeMs <= 0) &&
        (totalTimeAllowed <= 0) &&
        (nodeCount >= Config::Network.SelfPlay.NumSimulations))
    {
        workCoordinator->OnWorkItemCompleted();
        return;
    }
}

void SelfPlayWorker::PrintPrincipleVariation(bool searchFinished)
{
    const Node* root = _games[0].Root();
    std::vector<Move> principleVariation;

    const Node* bestChild = root->bestChild.load(std::memory_order_relaxed);
    if (!bestChild)
    {
        // No best move was found, so this is either a terminal node (mate or draw-on-the-board)
        // or not enough nodes have been explored, in which case we take max prior if explored,
        // or just MOVE_NONE otherwise. Only print for finished searches: don't spam before
        // finding bestChild normally.
        if (searchFinished)
        {
            const int rootEitherMateN = root->terminalValue.load(std::memory_order_relaxed).EitherMateN();
            std::cout << "info depth 0" << ((rootEitherMateN != 0) ? " score mate 0" : " score cp 0") << std::endl;
        }
        return;
    }

    const Node* pvBestChild = bestChild;
    while (pvBestChild)
    {
        principleVariation.push_back(Move(pvBestChild->move));
        pvBestChild = pvBestChild->bestChild.load(std::memory_order_relaxed);
    }

    const bool debug = _searchState->debug.load(std::memory_order_relaxed);
    auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::duration sinceSearchStart = (now - _searchState->searchStart);
    _searchState->lastPrincipleVariationPrint = now;

    // Value is from the parent's perspective, so that's already correct for the root perspective
    const int eitherMateN = bestChild->terminalValue.load(std::memory_order_relaxed).EitherMateN();
    const float value = bestChild->Value();
    const int depth = static_cast<int>(principleVariation.size());
    const int64_t searchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(sinceSearchStart).count();
    const int nodeCount = _searchState->nodeCount.load(std::memory_order_relaxed);
    const int tablebaseHitCount = _searchState->tablebaseHitCount.load(std::memory_order_relaxed);
    const float searchTimeSeconds = std::chrono::duration<float>(sinceSearchStart).count();
    const int nodesPerSecond = static_cast<int>(nodeCount / searchTimeSeconds);
    const int failedNodesPerSecond = static_cast<int>(_searchState->failedNodeCount.load(std::memory_order_relaxed) / searchTimeSeconds);
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

    std::cout << " nodes " << nodeCount << " nps " << nodesPerSecond;
    if (debug)
    {
        std::cout << " fnps " << failedNodesPerSecond;
    }
    std::cout << " tbhits " << tablebaseHitCount << " time " << searchTimeMs << " hashfull " << hashfullPermille;
    if (debug)
    {
        std::cout << " hashhit " << PredictionCache::Instance.PermilleHits()
            << " hashevict " << PredictionCache::Instance.PermilleEvictions();
    }
    std::cout << " pv";
    for (Move move : principleVariation)
    {
        std::cout << " " << UCI::move(move, false /* chess960 */);
    }
    std::cout << std::endl;
}

void SelfPlayWorker::SearchInitialize(const SelfPlayGame* position)
{
    // Set up parallelism. Make N games share a tree but have their own image/value/policy slots.
    _currentParallelism = 0;
    for (int i = 0; i < _games.size(); i++)
    {
        ClearGame(i);
        _states[i] = _states[0];
        _gameStarts[i] = _gameStarts[0];
        _games[i] = position->SpawnShadow(&_images[i], &_values[i], &_policies[i]);
    }
}

void SelfPlayWorker::CommentOnPosition(INetwork* network)
{
    std::unique_ptr<INetwork::CommentaryInputPlanes> image(std::make_unique<INetwork::CommentaryInputPlanes>());

    _games[0].GenerateCommentaryImage(image->data());
    const std::vector<std::string> comments = network->PredictCommentaryBatch(1, image.get());
    std::cout << comments[0] << std::endl;
}