#include "Game.h"

#include <Stockfish/thread.h>

#include "Config.h"

int Game::QueenKnightPlane[SQUARE_NB];
Key Game::PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];
std::array<float, 25> Game::PuctMateTerm;
thread_local PoolAllocator<StateInfo, Game::BlockSizeBytes> Game::StateAllocator;

StateInfo* Game::AllocateState()
{
    return reinterpret_cast<StateInfo*>(StateAllocator.Allocate());
}

void Game::FreeState(StateInfo* state)
{
    StateAllocator.Free(state);
}

void Game::Initialize()
{
    // Set up mappings for queen and knight moves to index into policy planes.

    for (int& plane : QueenKnightPlane)
    {
        plane = NO_PLANE;
    }
    int nextPlane = 0;

    const Direction QueenDirections[] = { NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST };
    const int MaxDistance = 7;
    for (Direction direction : QueenDirections)
    {
        for (int distance = 1; distance <= MaxDistance; distance++)
        {
            QueenKnightPlane[(SQUARE_NB + direction * distance) % SQUARE_NB] = nextPlane++;
        }
    }

    const int KnightMoves[] = { NORTH_EAST + NORTH, NORTH_EAST + EAST, SOUTH_EAST + EAST, SOUTH_EAST + SOUTH,
        SOUTH_WEST + SOUTH, SOUTH_WEST + WEST, NORTH_WEST + WEST, NORTH_WEST + NORTH };
    for (int delta : KnightMoves)
    {
        QueenKnightPlane[(SQUARE_NB + delta) % SQUARE_NB] = nextPlane++;
    }

    // Set up additional Zobrist hash keys for prediction caching (additional info beyond position).
    PRNG rng(7607098); // Arbitrary seed
    for (int i = 0; i <= NoProgressSaturationCount; i++)
    {
        PredictionCache_NoProgressCount[i] = rng.rand<Key>();
    }

    // Set up scores for various mate-in-N to encourage visits and differentiate depths.
    PuctMateTerm[0] = 0.f;
    PuctMateTerm[1] = 1.f;
    for (int n = 2; n < PuctMateTerm.size(); n++)
    {
        PuctMateTerm[n] = (1.f / (1 << n));
    }
    assert(PuctMateTerm[2] == 0.25f);
    assert(PuctMateTerm[3] == 0.125f);
}

Game::Game()
    : _parentState(nullptr)
    , _currentState(AllocateState())
    , _previousPositions(INetwork::InputPreviousPositionCount)
    , _previousPositionsOldest(0)

{
    _position.set(Config::StartingPosition, false /* isChess960 */, _currentState, Threads.main());
}

Game::Game(const std::string& fen, const std::vector<Move>& moves)
    : _parentState(nullptr)
    , _currentState(AllocateState())
    , _previousPositions(INetwork::InputPreviousPositionCount)
    , _previousPositionsOldest(0)

{
    _position.set(fen, false /* isChess960 */, _currentState, Threads.main());

    for (Move move : moves)
    {
        ApplyMove(move);
    }
}

Game::Game(const Game& other)
    : _position(other._position)
    , _parentState(other._currentState) // Don't delete the parent game's states.
    , _currentState(other._currentState)
    , _previousPositions(other._previousPositions)
    , _previousPositionsOldest(other._previousPositionsOldest)
{
    assert(&other != this);
}

Game& Game::operator=(const Game& other)
{
    assert(&other != this);

    // Free existing allocations before assigning.
    Free();

    _position = other._position;
    _parentState = other._currentState; // Don't delete the parent game's states.
    _currentState = other._currentState;
    _previousPositions = other._previousPositions;
    _previousPositionsOldest = other._previousPositionsOldest;

    return *this;
}

Game::Game(Game&& other) noexcept
    : _position(other._position)
    , _parentState(other._parentState)
    , _currentState(other._currentState)
    , _previousPositions(std::move(other._previousPositions))
    , _previousPositionsOldest(other._previousPositionsOldest)
{
    other._parentState = nullptr;
    other._currentState = nullptr;

    assert(&other != this);
}

Game& Game::operator=(Game&& other) noexcept
{
    assert(&other != this);

    // Free existing allocations before assigning.
    Free();

    _position = other._position;
    _parentState = other._parentState;
    _currentState = other._currentState;
    _previousPositions = std::move(other._previousPositions);
    _previousPositionsOldest = other._previousPositionsOldest;

    other._parentState = nullptr;
    other._currentState = nullptr;

    return *this;
}

Game::~Game()
{
    Free();
}

Color Game::ToPlay() const
{
    return _position.side_to_move();
}

void Game::ApplyMove(Move move)
{
    _previousPositions[_previousPositionsOldest] = _position;
    _previousPositionsOldest = (_previousPositionsOldest + 1) % INetwork::InputPreviousPositionCount;

    _currentState = AllocateState();
    _position.do_move(move, *_currentState);
}

void Game::ApplyMoveMaybeNull(Move move)
{
    _previousPositions[_previousPositionsOldest] = _position;
    _previousPositionsOldest = (_previousPositionsOldest + 1) % INetwork::InputPreviousPositionCount;

    _currentState = AllocateState();
    if (move != MOVE_NULL)
    {
        _position.do_move(move, *_currentState);
    }
    else
    {
        _position.do_null_move(*_currentState);
    }
}

Move Game::ApplyMoveInfer(const INetwork::PackedPlane* resultingPieces)
{
    StateInfo state;
    const MoveList legalMoves = MoveList<LEGAL>(_position);
    INetwork::PackedPlane checkPieces[INetwork::InputPiecePlanesPerPosition];

    for (const Move move : legalMoves)
    {
        _position.do_move(move, state);
        GeneratePiecePlanes(checkPieces, 0, _position);
        if (PiecesMatch(checkPieces, resultingPieces))
        {
            _position.undo_move(move);
            ApplyMove(move);
            return move;
        }
        _position.undo_move(move);
    }
    throw std::runtime_error("Impossible to reach provided position via legal move");
}

Move Game::ApplyMoveGuess(float result, const std::map<Move, float>& policy)
{
    // Walk through legal moves from highest value to lowest and pick the first one that matches the game result.
    // We don't have enough information here to fully replicate "SelfPlayWorker::WorseThan" logic, e.g. "TerminalValue"
    // and mate-proving, but hopefully at MCTS time that guided the "bestNode" to the highest value in time.
    //
    // This guess can definitely be wrong though and shouldn't be trusted too much. It's irrelevant for training and
    // is mainly to allow sane PGN generation for the debug GUI.
    std::vector<std::pair<Move, float>> bestMoves(policy.begin(), policy.end());
    std::sort(bestMoves.begin(), bestMoves.end(), [](const auto& a, const auto& b) { return  (a.second > b.second); });
    StateInfo state;
    for (const auto& [move, value] : bestMoves)
    {
        _position.do_move(move, state);
        const MoveList legalMoves = MoveList<LEGAL>(_position);
        const float moveResult = FlipValue(ToPlay(),
            (legalMoves.size() == 0) ?
                (_position.checkers() ? CHESSCOACH_VALUE_LOSS : CHESSCOACH_VALUE_DRAW) :
                IsDrawByNoProgressOrThreefoldRepetition() ?
                    CHESSCOACH_VALUE_DRAW :
                    (Ply() >= Config::Network.SelfPlay.MaxMoves) ?
                        CHESSCOACH_VALUE_DRAW :
                        CHESSCOACH_VALUE_UNINITIALIZED);
        if (moveResult == result)
        {
            _position.undo_move(move);
            ApplyMove(move);
            return move;
        }
        _position.undo_move(move);
    }
    throw std::runtime_error("Failed to guess move matching result and policy");
}

// Avoid Position::is_draw because it regenerates legal moves.
// If we've already just checked for checkmate and stalemate then this works fine.
bool Game::IsDrawByNoProgressOrThreefoldRepetition() const
{
    const StateInfo* stateInfo = _position.state_info();

    return
        // Omit "and not checkmate" from Position::is_draw.
        (stateInfo->rule50 > 99) ||
        // Stockfish encodes 3-repetition as negative.
        (stateInfo->repetition < 0);
}

bool Game::PiecesMatch(const INetwork::PackedPlane* a, const INetwork::PackedPlane* b) const
{
    for (int i = 0; i < INetwork::InputPiecePlanesPerPosition; i++)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

int Game::Ply() const
{
    return _position.game_ply();
}

Key Game::GenerateImageKey() const
{
    // No need to flip anything for hash keys: for a particular position, it's always the same player to move.
    // Stockfish key includes material and castling rights, so we just need to add history and no-progress.
    // Because Zobrist keys are iteratively updated via moves, they collide when combining positions that are
    // reached by different permutations of the same set of moves. So, rotate the key to differentiate.

    // Add no-progress plane to key.
    Key key = PredictionCache_NoProgressCount[std::min(NoProgressSaturationCount, _position.rule50_count())];

    // Add last 7 positions.
    int previousPositionIndex = _previousPositionsOldest;
    for (int i = 0; i < INetwork::InputPreviousPositionCount; i++)
    {
        const Position& position = _previousPositions[previousPositionIndex];
        key ^= (position.state_info() ? Rotate(position.key(), INetwork::InputPreviousPositionCount - i) : 0);
        previousPositionIndex = (previousPositionIndex + 1) % INetwork::InputPreviousPositionCount;
    }

    // Add current position.
    key ^= _position.key();

    return key;
}

void Game::GenerateImage(INetwork::InputPlanes& imageOut) const
{
    GenerateImage(imageOut.data());
}

void Game::GenerateImage(INetwork::PackedPlane* imageOut) const
{
    int nextPlane = 0;

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add last 7 positions' pieces, planes 0-83.
    assert(nextPlane == 0);
    int previousPositionIndex = _previousPositionsOldest;
    for (int i = 0; i < INetwork::InputPreviousPositionCount; i++)
    {
        const Position& position = _previousPositions[previousPositionIndex];
        GeneratePiecePlanes(imageOut, nextPlane, position);

        nextPlane += INetwork::InputPiecePlanesPerPosition;
        previousPositionIndex = (previousPositionIndex + 1) % INetwork::InputPreviousPositionCount;
    }

    // Add current position's pieces, planes 84-95.
    assert(nextPlane == 84);
    GeneratePiecePlanes(imageOut, nextPlane, _position);
    nextPlane += INetwork::InputPiecePlanesPerPosition;

    // Castling planes 96-99
    assert(nextPlane == 96);
    FillPlane(imageOut[nextPlane++], _position.can_castle(toPlay & KING_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(toPlay & QUEEN_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(~toPlay & KING_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(~toPlay & QUEEN_SIDE));

    // No-progress plane 100
    // This is a special case, not to be bit-unpacked, but instead interpreted as an integer
    // to be normalized via the no-progress saturation count (99).
    assert(nextPlane == 100);
    imageOut[nextPlane++] = _position.rule50_count();

    static_assert(INetwork::InputPreviousPositionCount == 7);
    static_assert(INetwork::InputPiecePlanesPerPosition == 12);
    static_assert(INetwork::InputAuxiliaryPlaneCount == 5);
    static_assert(INetwork::InputPlaneCount == 101);
    assert(nextPlane == INetwork::InputPlaneCount);
}

void Game::GenerateImageCompressed(INetwork::PackedPlane* piecesOut, INetwork::PackedPlane* auxiliaryOut) const
{
    int nextPieces = 0;

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add current position's pieces, planes 84-95 (becoming future positions' planes 72-83, 60-71, etc.).
    GeneratePiecePlanes(piecesOut, nextPieces, _position);
    nextPieces += INetwork::InputPiecePlanesPerPosition;

    // Castling planes 96-99
    int nextAuxiliary = 0;
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(toPlay & KING_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(toPlay & QUEEN_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(~toPlay & KING_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(~toPlay & QUEEN_SIDE));

    // No-progress plane 100
    // This is a special case, not to be bit-unpacked, but instead interpreted as an integer
    // to be normalized via the no-progress saturation count (99).
    auxiliaryOut[nextAuxiliary++] = _position.rule50_count();

    assert(nextPieces == INetwork::InputPiecePlanesPerPosition);
    assert(nextAuxiliary == INetwork::InputAuxiliaryPlaneCount);
}

float& Game::PolicyValue(INetwork::OutputPlanes& policyInOut, Move move) const
{
    return PolicyValue(reinterpret_cast<INetwork::PlanesPointer>(policyInOut.data()), move);
}

float& Game::PolicyValue(INetwork::PlanesPointerFlat policyInOut, Move move) const
{
    return PolicyValue(reinterpret_cast<INetwork::PlanesPointer>(policyInOut), move);
}

float& Game::PolicyValue(INetwork::PlanesPointer policyInOut, Move move) const
{
    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    move = FlipMove(ToPlay(), move);
    Square from = from_sq(move);
    Square to = to_sq(move);

    int plane;
    PieceType promotion;
    if ((type_of(move) == PROMOTION) && ((promotion = promotion_type(move)) != QUEEN))
    {
        plane = UnderpromotionPlane[promotion - KNIGHT][to - from - NORTH_WEST];
        assert((plane >= 0) && (plane < INetwork::OutputPlaneCount));
    }
    else
    {
        plane = QueenKnightPlane[(to - from + SQUARE_NB) % SQUARE_NB];
        assert((plane >= 0) && (plane < INetwork::OutputPlaneCount));
    }

    return policyInOut[plane][rank_of(from)][file_of(from)];
}

// Callers must zero "policyOut" before calling: only some values are set.
void Game::GeneratePolicy(const std::map<Move, float>& childVisits, INetwork::OutputPlanes& policyOut) const
{
    for (const auto pair : childVisits)
    {
        PolicyValue(policyOut, pair.first) = pair.second;
    }
}

// Callers must zero "policyValuesOut" before calling: only some values are set.
void Game::GeneratePolicyCompressed(const std::map<Move, float>& childVisits, int64_t* policyIndicesOut, float* policyValuesOut) const
{
    int i = 0;
    for (const auto& [move, value] : childVisits)
    {
        // Let "PolicyValue" generate a flat index via [plane][rank][file] and just never dereference.
        const INetwork::PlanesPointerFlat zero = 0;
        intptr_t distance = (&PolicyValue(zero, move) - zero);
        policyIndicesOut[i] = static_cast<int>(distance);
        policyValuesOut[i] = value;
        i++;
    }
}

// Requires that the caller has zero-initialized "policyOut".
void Game::GeneratePolicyDecompress(int childVisitsSize, const int64_t* policyIndices, const float* policyValues, INetwork::OutputPlanes& policyOut)
{
    INetwork::PlanesPointerFlat policyFlat = reinterpret_cast<INetwork::PlanesPointerFlat>(policyOut.data());
    for (int i = 0; i < childVisitsSize; i++)
    {
        policyFlat[policyIndices[i]] = policyValues[i];
    }
}

const Position& Game::GetPosition() const
{
    return _position;
}

void Game::Free()
{
    // Delete any states allocated purely in this game branch, but not any in the parent
    // (i.e. when created by copy constructor/copy assignment and _parentState is not null).
    StateInfo* current = _currentState;
    while (current != _parentState)
    {
        assert(current);
        StateInfo* free = current;
        current = current->previous;
        FreeState(free);
    }

    _currentState = nullptr;
    _parentState = nullptr;

    // Nodes are freed outside of Game objects because they outlive the games through MCTS tree reuse.
}

void Game::GeneratePiecePlanes(INetwork::PackedPlane* imageOut, int planeOffset, const Position& position) const
{
    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = position.side_to_move();

    imageOut[planeOffset + 0] = position.pieces(toPlay, PAWN);
    imageOut[planeOffset + 1] = position.pieces(toPlay, KNIGHT);
    imageOut[planeOffset + 2] = position.pieces(toPlay, BISHOP);
    imageOut[planeOffset + 3] = position.pieces(toPlay, ROOK);
    imageOut[planeOffset + 4] = position.pieces(toPlay, QUEEN);
    imageOut[planeOffset + 5] = position.pieces(toPlay, KING);

    imageOut[planeOffset + 6] = position.pieces(~toPlay, PAWN);
    imageOut[planeOffset + 7] = position.pieces(~toPlay, KNIGHT);
    imageOut[planeOffset + 8] = position.pieces(~toPlay, BISHOP);
    imageOut[planeOffset + 9] = position.pieces(~toPlay, ROOK);
    imageOut[planeOffset + 10] = position.pieces(~toPlay, QUEEN);
    imageOut[planeOffset + 11] = position.pieces(~toPlay, KING);

    static_assert(INetwork::InputPiecePlanesPerPosition == 12);

    // Piece colors are already conditionally flipped via toPlay/~toPlay ordering. Flip all vertically if BLACK to play.
    if (toPlay == BLACK)
    {
        for (int i = planeOffset; i < planeOffset + INetwork::InputPiecePlanesPerPosition; i++)
        {
            imageOut[i] = FlipBoard(imageOut[i]);
        }
    }
}

void Game::FillPlane(INetwork::PackedPlane& plane, bool value) const
{
    plane = FillPlanePacked[static_cast<int>(value)];
}

Key Game::Rotate(Key key, unsigned int distance) const
{
    // https://stackoverflow.com/questions/776508/best-practices-for-circular-shift-rotate-operations-in-c
    // https://blog.regehr.org/archives/1063
    
    static_assert(std::is_same<uint64_t, Key>::value);
    
    const unsigned int mask = 63;
    distance &= mask;
    return (key >> distance) | (key << (-distance & mask));
}