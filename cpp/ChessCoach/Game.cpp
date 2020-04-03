#include "Game.h"

#include <Stockfish/thread.h>

#include "Config.h"

int Game::QueenKnightPlane[SQUARE_NB];

float Game::FlipValue(Color toPlay, float value)
{
    return (toPlay == WHITE) ? value : FlipValue(value);
}

float Game::FlipValue(float value)
{
    return (CHESSCOACH_VALUE_WIN - value);
}

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
}

Game::Game()
    : _positionStates(new std::deque<StateInfo>(1))
    , _previousMoves(INetwork::InputPreviousMoveCount) // Fills with MOVE_NONE == 0
    , _previousMovesOldest(0)

{
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    _position.set(StartFEN, false /* isChess960 */, &_positionStates->back(), Threads.main());
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
    , _previousMoves(std::move(other._previousMoves))
    , _previousMovesOldest(other._previousMovesOldest)
{
    assert(&other != this);
}

Game& Game::operator=(Game&& other) noexcept
{
    assert(&other != this);

    _position = other._position;
    _positionStates = std::move(other._positionStates);
    _previousMoves = std::move(other._previousMoves);
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
    _previousMovesOldest = (_previousMovesOldest + 1) % INetwork::InputPreviousMoveCount;
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
    for (int i = 0; i < INetwork::InputPreviousMoveCount; i++)
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
        previousMoveIndex = (previousMoveIndex + 1) % INetwork::InputPreviousMoveCount;
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
    const float normalizedFiftyRuleCount = _position.rule50_count() / 99.0f;
    FillPlane(image[24], normalizedFiftyRuleCount);

    return image;
}

INetwork::OutputPlanes Game::GeneratePolicy(const std::unordered_map<Move, float>& childVisits) const
{
    INetwork::OutputPlanes policy = {};

    for (auto pair : childVisits)
    {
        PolicyValue(policy, pair.first) = pair.second;
    }

    return policy;
}
#pragma warning(default:6262) // Ignore stack warning, caller can emplace to heap via RVO.

void Game::FillPlane(INetwork::Plane& plane, float value) const
{
    std::fill(&plane[0][0], &plane[0][0] + INetwork::PlaneFloatCount, value);
}