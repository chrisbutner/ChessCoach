#include "Game.h"

#include <Stockfish/thread.h>
#include <Stockfish/evaluate.h>

#include "Config.h"

int Game::QueenKnightPlane[SQUARE_NB];
Key Game::PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];
std::array<float, 25> Game::UcbMateTerm;

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
    : _positionStates(new std::deque<StateInfo>(1))
    , _previousPositions(INetwork::InputPreviousPositionCount)
    , _previousPositionsOldest(0)

{
    _position.set(Config::StartingPosition, false /* isChess960 */, &_positionStates->back(), Threads.main());
}

Game::Game(const std::string& fen, const std::vector<Move>& moves)
    : _positionStates(new std::deque<StateInfo>(1))
    , _previousPositions(INetwork::InputPreviousPositionCount)
    , _previousPositionsOldest(0)

{
    _position.set(fen, false /* isChess960 */, &_positionStates->back(), Threads.main());

    for (Move move : moves)
    {
        ApplyMove(move);
    }
}

Game::Game(const Game& other)
    : _position(other._position)
    , _positionStates(new std::deque<StateInfo>())
    , _previousPositions(other._previousPositions)
    , _previousPositionsOldest(other._previousPositionsOldest)
{
    assert(&other != this);
}

Game& Game::operator=(const Game& other)
{
    assert(&other != this);

    _position = other._position;
    _positionStates.reset(new std::deque<StateInfo>());
    _previousPositions = other._previousPositions;
    _previousPositionsOldest = other._previousPositionsOldest;

    return *this;
}

Game::Game(Game&& other) noexcept
    : _position(other._position)
    , _positionStates(std::move(other._positionStates))
    , _previousPositions(std::move(other._previousPositions))
    , _previousPositionsOldest(other._previousPositionsOldest)
{
    assert(&other != this);
}

Game& Game::operator=(Game&& other) noexcept
{
    assert(&other != this);

    _position = other._position;
    _positionStates = std::move(other._positionStates);
    _previousPositions = std::move(other._previousPositions);
    _previousPositionsOldest = other._previousPositionsOldest;

    return *this;
}

Game::~Game()
{
}

Color Game::ToPlay() const
{
    return _position.side_to_move();
}

void Game::ApplyMove(Move move)
{
    _previousPositions[_previousPositionsOldest] = _position;
    _previousPositionsOldest = (_previousPositionsOldest + 1) % INetwork::InputPreviousPositionCount;

    _positionStates->emplace_back();
    _position.do_move(move, _positionStates->back());
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
    INetwork::InputPlanes image = {};
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
    if (_position.can_castle(toPlay & KING_SIDE))
    {
        FillPlane(image[nextPlane], 1.f);
    }
    nextPlane++;
    if (_position.can_castle(toPlay & QUEEN_SIDE))
    {
        FillPlane(image[nextPlane], 1.f);
    }
    nextPlane++;
    if (_position.can_castle(~toPlay & KING_SIDE))
    {
        FillPlane(image[nextPlane], 1.f);
    }
    nextPlane++;
    if (_position.can_castle(~toPlay & QUEEN_SIDE))
    {
        FillPlane(image[nextPlane], 1.f);
    }
    nextPlane++;

    // No-progress plane 100
    assert(nextPlane == 100);
    const float normalizedFiftyRuleCount = std::min(1.f, static_cast<float>(_position.rule50_count()) / NoProgressSaturationCount);
    FillPlane(image[nextPlane++], normalizedFiftyRuleCount);

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

void Game::GeneratePiecePlanes(INetwork::InputPlanes& image, int planeOffset, const Position& position) const
{
    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = position.side_to_move();

    for (Rank rank = RANK_1; rank <= RANK_8; ++rank)
    {
        for (File file = FILE_A; file <= FILE_H; ++file)
        {
            const Piece piece = FlipPiece[toPlay][position.piece_on(FlipSquare(toPlay, make_square(file, rank)))];
            const int plane = ImagePiecePlane[piece];
            if (plane != NO_PLANE)
            {
                // If any auxilary info is added to position planes then they won't both be 12 anymore.
                const int piecePlanesPerPosition = 12;
                static_assert(piecePlanesPerPosition <= INetwork::InputPlanesPerPosition);
                assert((plane >= 0) && (plane < piecePlanesPerPosition));
                image[planeOffset + plane][rank][file] = 1.f;
            }
        }
    }
}

void Game::FillPlane(INetwork::Plane& plane, float value) const
{
    std::fill(&plane[0][0], &plane[0][0] + INetwork::PlaneFloatCount, value);
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