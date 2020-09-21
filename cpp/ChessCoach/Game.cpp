#include "Game.h"

#include <Stockfish/thread.h>
#include <Stockfish/evaluate.h>

#include "Config.h"

int Game::QueenKnightPlane[SQUARE_NB];
Key Game::PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];
std::array<float, 25> Game::UcbMateTerm;
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
    UcbMateTerm[0] = 0.f;
    UcbMateTerm[1] = 1.f;
    for (int n = 2; n < UcbMateTerm.size(); n++)
    {
        UcbMateTerm[n] = (1.f / (1 << n));
    }
    assert(UcbMateTerm[2] == 0.25f);
    assert(UcbMateTerm[3] == 0.125f);
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

int Game::Ply() const
{
    return _position.game_ply();
}

float& Game::PolicyValue(INetwork::OutputPlanes& policy, Move move) const
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

    return policy[plane][rank_of(from)][file_of(from)];
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

#pragma warning(disable:6262) // Ignore stack warning, caller can emplace to heap via RVO.
INetwork::InputPlanes Game::GenerateImage() const
{
    INetwork::InputPlanes image;
    int nextPlane = 0;

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add last 7 positions' pieces, planes 0-83.
    assert(nextPlane == 0);
    int previousPositionIndex = _previousPositionsOldest;
    for (int i = 0; i < INetwork::InputPreviousPositionCount; i++)
    {
        const Position& position = _previousPositions[previousPositionIndex];
        GeneratePiecePlanes(image, nextPlane, position);

        nextPlane += INetwork::InputPlanesPerPosition;
        previousPositionIndex = (previousPositionIndex + 1) % INetwork::InputPreviousPositionCount;
    }

    // Add current position's pieces, planes 84-95.
    assert(nextPlane == 84);
    GeneratePiecePlanes(image, nextPlane, _position);
    nextPlane += INetwork::InputPlanesPerPosition;

    // Castling planes 96-99
    assert(nextPlane == 96);
    FillPlane(image[nextPlane++], _position.can_castle(toPlay & KING_SIDE));
    FillPlane(image[nextPlane++], _position.can_castle(toPlay & QUEEN_SIDE));
    FillPlane(image[nextPlane++], _position.can_castle(~toPlay & KING_SIDE));
    FillPlane(image[nextPlane++], _position.can_castle(~toPlay & QUEEN_SIDE));

    // No-progress plane 100
    // This is a special case, not to be bit-unpacked, but instead interpreted as an integer
    // to be normalized via the no-progress saturation count (99).
    assert(nextPlane == 100);
    image[nextPlane++] = _position.rule50_count();

    static_assert(INetwork::InputPreviousPositionCount == 7);
    static_assert(INetwork::InputPlanesPerPosition == 12);
    static_assert(INetwork::InputPlaneCount == 101);
    assert(nextPlane == INetwork::InputPlaneCount);
    return image;
}

INetwork::OutputPlanes Game::GeneratePolicy(const std::map<Move, float>& childVisits) const
{
    INetwork::OutputPlanes policy = {};

    for (auto pair : childVisits)
    {
        PolicyValue(policy, pair.first) = pair.second;
    }

    return policy;
}
#pragma warning(default:6262) // Ignore stack warning, caller can emplace to heap via RVO.

bool Game::StockfishCanEvaluate() const
{
    return !_position.checkers();
}

float Game::StockfishEvaluation() const
{
    const Value centipawns = Eval::evaluate(_position);
    const float probability = CentipawnsToProbability(static_cast<float>(centipawns));
    return probability;
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

void Game::GeneratePiecePlanes(INetwork::InputPlanes& image, int planeOffset, const Position& position) const
{
    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = position.side_to_move();

    image[planeOffset + 0] = position.pieces(toPlay, PAWN);
    image[planeOffset + 1] = position.pieces(toPlay, KNIGHT);
    image[planeOffset + 2] = position.pieces(toPlay, BISHOP);
    image[planeOffset + 3] = position.pieces(toPlay, ROOK);
    image[planeOffset + 4] = position.pieces(toPlay, QUEEN);
    image[planeOffset + 5] = position.pieces(toPlay, KING);

    image[planeOffset + 6] = position.pieces(~toPlay, PAWN);
    image[planeOffset + 7] = position.pieces(~toPlay, KNIGHT);
    image[planeOffset + 8] = position.pieces(~toPlay, BISHOP);
    image[planeOffset + 9] = position.pieces(~toPlay, ROOK);
    image[planeOffset + 10] = position.pieces(~toPlay, QUEEN);
    image[planeOffset + 11] = position.pieces(~toPlay, KING);

    // Piece colors are already conditionally flipped via toPlay/~toPlay ordering. Flip all vertically if BLACK to play.
    if (toPlay == BLACK)
    {
        for (int i = planeOffset; i < planeOffset + 12; i++)
        {
            image[i] = FlipBoard(image[i]);
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