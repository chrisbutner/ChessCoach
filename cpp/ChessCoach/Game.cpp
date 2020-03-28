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
{
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    _position.set(StartFEN, false /* isChess960 */, &_positionStates->back(), Threads.main());
}

Game::Game(const Game& other)
    : _position(other._position)
    , _positionStates(new std::deque<StateInfo>())
{
    assert(&other != this);
}

Game& Game::operator=(const Game& other)
{
    assert(&other != this);

    _position = other._position;
    _positionStates.reset(new std::deque<StateInfo>());

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
}

int Game::Ply() const
{
    return _position.game_ply();
}

float& Game::PolicyValue(OutputPlanesPtr policy, Move move) const
{
    // If it's black to play, rotate the board and flip colors: always from the "current player's" perspective.
    Square from = RotateSquare(ToPlay(), from_sq(move));
    Square to = RotateSquare(ToPlay(), to_sq(move));

    int plane;
    PieceType promotion;
    if ((type_of(move) == PROMOTION) && ((promotion = promotion_type(move)) != QUEEN))
    {
        plane = UnderpromotionPlane[promotion - KNIGHT][to - from - NORTH_WEST];
        assert((plane >= 0) && (plane < 73));
    }
    else
    {
        plane = QueenKnightPlane[(to - from + SQUARE_NB) % SQUARE_NB];
        assert((plane >= 0) && (plane < 73));
    }

    return policy[plane][rank_of(from)][file_of(from)];
}

#pragma warning(disable:6262) // Ignore stack warning, caller can emplace to heap via RVO.
InputPlanes Game::GenerateImage() const
{
    InputPlanes image = {};

    // If it's black to play, rotate the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();
    for (Rank rank = RANK_1; rank <= RANK_8; ++rank)
    {
        for (File file = FILE_A; file <= FILE_H; ++file)
        {
            Piece piece = FlipPiece[toPlay][_position.piece_on(RotateSquare(toPlay, make_square(file, rank)))];
            int plane = ImagePiecePlane[piece];
            if (plane != NO_PLANE)
            {
                assert((plane >= 0) && (plane < 12));
                image[plane][rank][file] = 1.f;
            }
        }
    }

    return image;
}

OutputPlanes Game::GeneratePolicy(const std::unordered_map<Move, float>& childVisits) const
{
    OutputPlanes policy = {};
    OutputPlanesPtr policyPtr = reinterpret_cast<float(*)[8][8]>(policy.data());

    const Color toPlay = ToPlay();
    for (auto pair : childVisits)
    {
        PolicyValue(policyPtr, pair.first) = pair.second;
    }

    return policy;
}
#pragma warning(default:6262) // Ignore stack warning, caller can emplace to heap via RVO.