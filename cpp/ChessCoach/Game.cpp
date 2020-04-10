#include "Game.h"

#include <Stockfish/thread.h>
#include <Stockfish/evaluate.h>

#include "Config.h"

int Game::QueenKnightPlane[SQUARE_NB];
Key Game::PredictionCache_PreviousMoveSquare[Config::InputPreviousMoveCount][SQUARE_NB];
Key Game::PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];

void Game::Initialize()
{
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

    PRNG rng(7607098); // Arbitrary seed
    for (int i = 0; i < Config::InputPreviousMoveCount; i++)
    {
        for (Square square = SQ_A1; square <= SQ_H8; ++square)
        {
            PredictionCache_PreviousMoveSquare[i][square] = rng.rand<Key>();
        }
    }
    for (int i = 0; i <= NoProgressSaturationCount; i++)
    {
        PredictionCache_NoProgressCount[i] = rng.rand<Key>();
    }
}

Game::Game()
    : _positionStates(new std::deque<StateInfo>(1))
    , _previousMoves{} // Fills with MOVE_NONE == 0
    , _previousMovesOldest(0)

{
    _position.set(Config::StartingPosition, false /* isChess960 */, &_positionStates->back(), Threads.main());
}

Game::Game(const std::string& fen)
    : _positionStates(new std::deque<StateInfo>(1))
    , _previousMoves{} // Fills with MOVE_NONE == 0
    , _previousMovesOldest(0)

{
    _position.set(fen, false /* isChess960 */, &_positionStates->back(), Threads.main());
}

Game::Game(const Game& other)
    : _position(other._position)
    , _positionStates(new std::deque<StateInfo>())
    , _previousMoves(other._previousMoves)
    , _previousMovesOldest(other._previousMovesOldest)
{
    assert(&other != this);
}

Game& Game::operator=(const Game& other)
{
    assert(&other != this);

    _position = other._position;
    _positionStates.reset(new std::deque<StateInfo>());
    _previousMoves = other._previousMoves;
    _previousMovesOldest = other._previousMovesOldest;

    return *this;
}

Game::Game(Game&& other) noexcept
    : _position(other._position)
    , _positionStates(std::move(other._positionStates))
    , _previousMoves(other._previousMoves)
    , _previousMovesOldest(other._previousMovesOldest)
{
    assert(&other != this);
}

Game& Game::operator=(Game&& other) noexcept
{
    assert(&other != this);

    _position = other._position;
    _positionStates = std::move(other._positionStates);
    _previousMoves = other._previousMoves;
    _previousMovesOldest = other._previousMovesOldest;

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
    _positionStates->emplace_back();
    _position.do_move(move, _positionStates->back());
    _previousMoves[_previousMovesOldest] = move;
    _previousMovesOldest = (_previousMovesOldest + 1) % Config::InputPreviousMoveCount;
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
    // Stockfish key includes material (planes 0-11) and castling rights (planes 20-23).
    Key key = _position.key();

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add previous move planes 12-19 to key.
    int previousMoveIndex = _previousMovesOldest;
    for (int i = 0; i < Config::InputPreviousMoveCount; i++)
    {
        Move move = _previousMoves[previousMoveIndex];
        if (move != MOVE_NONE)
        {
            move = FlipMove(toPlay, move);
            Square from = from_sq(move);
            Square to = to_sq(move);
            key ^= PredictionCache_PreviousMoveSquare[i][from];
            key ^= PredictionCache_PreviousMoveSquare[i][to];
        }
        previousMoveIndex = (previousMoveIndex + 1) % Config::InputPreviousMoveCount;
    }

    // Add no-progress plane 24 to key.
    key ^= PredictionCache_NoProgressCount[std::min(NoProgressSaturationCount, _position.rule50_count())];

    return key;
}

#pragma warning(disable:6262) // Ignore stack warning, caller can emplace to heap via RVO.
INetwork::InputPlanes Game::GenerateImage() const
{
    INetwork::InputPlanes image = {};

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Piece planes 0-11
    for (Rank rank = RANK_1; rank <= RANK_8; ++rank)
    {
        for (File file = FILE_A; file <= FILE_H; ++file)
        {
            Piece piece = FlipPiece[toPlay][_position.piece_on(FlipSquare(toPlay, make_square(file, rank)))];
            int plane = ImagePiecePlane[piece];
            if (plane != NO_PLANE)
            {
                assert((plane >= 0) && (plane < InputPiecePlaneCount));
                image[plane][rank][file] = 1.f;
            }
        }
    }

    // Previous move planes 12-19
    int previousMoveIndex = _previousMovesOldest;
    for (int i = 0; i < Config::InputPreviousMoveCount; i++)
    {
        Move move = _previousMoves[previousMoveIndex];
        if (move != MOVE_NONE)
        {
            move = FlipMove(toPlay, move);
            Square from = from_sq(move);
            Square to = to_sq(move);
            image[i][rank_of(from)][file_of(from)] = 1.f;
            image[i][rank_of(to)][file_of(to)] = 1.f;
        }
        previousMoveIndex = (previousMoveIndex + 1) % Config::InputPreviousMoveCount;
    }

    // Castling planes 20-23
    if (_position.can_castle(KingsideRights[toPlay]))
    {
        FillPlane(image[20], 1.f);
    }
    if (_position.can_castle(KingsideRights[~toPlay]))
    {
        FillPlane(image[21], 1.f);
    }
    if (_position.can_castle(QueensideRights[toPlay]))
    {
        FillPlane(image[22], 1.f);
    }
    if (_position.can_castle(QueensideRights[~toPlay]))
    {
        FillPlane(image[23], 1.f);
    }

    // No-progress plane 24
    const float normalizedFiftyRuleCount = std::min(1.f, static_cast<float>(_position.rule50_count()) / NoProgressSaturationCount);
    FillPlane(image[24], normalizedFiftyRuleCount);

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

void Game::FillPlane(INetwork::Plane& plane, float value) const
{
    std::fill(&plane[0][0], &plane[0][0] + INetwork::PlaneFloatCount, value);
}