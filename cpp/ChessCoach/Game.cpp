// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#include "Game.h"

#include <Stockfish/thread.h>

#include "Config.h"

const float Game::CHESSCOACH_VALUE_SYZYGY_WIN = Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN - CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM);
const float Game::CHESSCOACH_VALUE_SYZYGY_DRAW = Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW);
const float Game::CHESSCOACH_VALUE_SYZYGY_LOSS = Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_LOSS + CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM);
int Game::QueenKnightPlane[256];
Key Game::PredictionCache_IsRepetition;
Key Game::PredictionCache_NoProgressCount[NoProgressSaturationCount + 1];
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
    assert(CHESSCOACH_VALUE_WIN > CHESSCOACH_VALUE_SYZYGY_WIN);
    assert(CHESSCOACH_VALUE_SYZYGY_WIN > CHESSCOACH_VALUE_SYZYGY_DRAW);
    assert(CHESSCOACH_VALUE_SYZYGY_DRAW == CHESSCOACH_VALUE_DRAW);
    assert(CHESSCOACH_VALUE_DRAW > CHESSCOACH_VALUE_SYZYGY_LOSS);
    assert(CHESSCOACH_VALUE_SYZYGY_LOSS > CHESSCOACH_VALUE_LOSS);

    // Set up mappings for queen and knight moves to index into policy planes.
    for (int& plane : QueenKnightPlane)
    {
        plane = NO_PLANE;
    }
    int nextPlane = 0;

    const Direction QueenDirections[] = { NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST };
    const Square QueenFrom[] = { SQ_A1, SQ_A8, SQ_H8, SQ_H1 };
    const int MaxDistance = 7;
    for (int directionIndex = 0; directionIndex < std::size(QueenDirections); directionIndex++)
    {
        const Direction direction = QueenDirections[directionIndex];
        const Square from = QueenFrom[directionIndex / 2];
        for (int distance = 1; distance <= MaxDistance; distance++)
        {
            QueenKnightPlane[Delta88(from, from + direction * distance)] = nextPlane++;
        }
    }

    const int KnightMoves[] = { NORTH_EAST + NORTH, NORTH_EAST + EAST, SOUTH_EAST + EAST, SOUTH_EAST + SOUTH,
        SOUTH_WEST + SOUTH, SOUTH_WEST + WEST, NORTH_WEST + WEST, NORTH_WEST + NORTH };
    const Square knightFrom = SQ_E4;
    for (int delta : KnightMoves)
    {
        QueenKnightPlane[Delta88(knightFrom, Square(knightFrom + delta))] = nextPlane++;
    }

    // Set up additional Zobrist hash keys for prediction caching (additional info beyond position).
    PRNG rng(7607098); // Arbitrary seed
    PredictionCache_IsRepetition = rng.rand<Key>();
    for (int i = 0; i <= NoProgressSaturationCount; i++)
    {
        PredictionCache_NoProgressCount[i] = rng.rand<Key>();
    }
}

Game::Game()
    : _parentState(nullptr)
    , _currentState(AllocateState())
    , _moves()
{
    _position.set(StartingPosition, false /* isChess960 */, _currentState, Threads.main());
}

Game::Game(const std::string& fen, const std::vector<Move>& moves)
    : _parentState(nullptr)
    , _currentState(AllocateState())
    , _moves() // Built up in each ApplyMove below
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
    , _moves(other._moves)
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
    _moves = other._moves;

    return *this;
}

Game::Game(Game&& other) noexcept
    : _position(other._position)
    , _parentState(other._parentState)
    , _currentState(other._currentState)
    , _moves(std::move(other._moves))
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
    _moves = std::move(other._moves);

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

bool Game::IsEndgame() const
{
    return ((_position.non_pawn_material()) <= Config::Network.SelfPlay.EndgameMaterialThreshold);
}

float Game::EndgameProportion() const
{
    const int material = std::clamp(_position.non_pawn_material(), EndgameLimit, MidgameLimit);
    return (static_cast<float>(MidgameLimit - material) / (MidgameLimit - EndgameLimit));
}

void Game::ApplyMove(Move move)
{
    _moves.push_back(move);
    _currentState = AllocateState();
    _position.do_move(move, *_currentState);
}

void Game::ApplyMoveMaybeNull(Move move)
{
    _moves.push_back(move);
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
    INetwork::PackedPlane checkPieces[INetwork::InputPieceAndRepetitionPlanesPerPosition];

    for (const Move move : legalMoves)
    {
        _position.do_move(move, state);
        GeneratePieceAndRepetitionPlanes(checkPieces, 0, _position, _position.side_to_move());
        if (PiecesAndRepetitionsMatch(checkPieces, resultingPieces))
        {
            _position.undo_move(move);
            ApplyMove(move);
            return move;
        }
        _position.undo_move(move);
    }
    throw ChessCoachException("Impossible to reach provided position via legal move");
}

Move Game::ApplyMoveInfer(const std::string& resultingFen)
{
    // When considering legal moves, it's enough to compare the presence of a piece on each square.
    // If one matches, none of the others can.
    const Game resulting(resultingFen, {});
    const Bitboard resultingPieces = resulting.GetPosition().pieces();

    StateInfo state;
    const MoveList legalMoves = MoveList<LEGAL>(_position);
    for (const Move move : legalMoves)
    {
        _position.do_move(move, state);
        if (_position.pieces() == resultingPieces)
        {
            _position.undo_move(move);
            ApplyMove(move);
            return move;
        }
        _position.undo_move(move);
    }
    throw ChessCoachException("Impossible to reach provided position via legal move");
}

Move Game::ApplyMoveGuess(float result, const std::map<Move, float>& policy)
{
    // Walk through legal moves from highest policy value to lowest and pick the first one that matches the game result.
    // We don't have enough information here to fully replicate "SelfPlayWorker::WorseThan" logic, e.g. "TerminalValue"
    // and mate-proving, but hopefully at MCTS time that guided the "bestNode" to the highest policy value in time.
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

    // No legal move matches the game result, so assume that the game was adjudicated and just pick the move with
    // the highest policy value as a wild guess.
    //
    // For "supervised" datasets, generated from PGNs rather than self-play tree statistics, policy values will be
    // one-hot, so the choice is easy.
    if (!bestMoves.empty())
    {
        const Move move = bestMoves[0].first;
        ApplyMove(move);
        return move;
    }

    throw ChessCoachException("No final move is possible in stored game");
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

bool Game::PiecesAndRepetitionsMatch(const INetwork::PackedPlane* a, const INetwork::PackedPlane* b) const
{
    for (int i = 0; i < INetwork::InputPieceAndRepetitionPlanesPerPosition; i++)
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

Key Game::GenerateImageKey(bool tryHard)
{
    // No need to flip anything for hash keys: for a particular position, it's always the same player to move,
    // side-to-move is encoded in the key, and we're not feeding in to a neural network, just differentiating.
    if (tryHard)
    {
        // In UCI mode (search/tournament/analysis), use the prediction cache as a transposition table such that
        // transpositions intentionally collide and reuse neural network predictions. This is a speed/accuracy
        // trade-off. Because the engine is effectively GPU/TPU-bound, with CPU to spare, getting a cache hit is
        // almost a 100% free node, which can be spent deeper inside the hit to get an even more accurate evaluation
        // in compensation. The risk is that (a) neural network volatility vs. history planes or (b) sharp edges
        // related to repetition or no-progress draw rules result in an overly optimistic or pessimistic evaluation
        // or skewed set of priors, resulting in wasting too many nodes or missing ideas. The hope is that the broader
        // and deeper search from the additional nodes more than compensates for any inaccuracies.
        //
        // In contrast, in self-play mode when generating training data (the "else" below), include history and no-progress
        // to match neural network planes and get as accurate a prediction as possible, at the expense of speed/throughput.
        // The idea there is to give as much information and gradient to the training process as possible, especially when
        // learning subtleties like 3-repetition and 50-move (no-progress) rules.
        //
        // It's possible that it would also be better to include less or no history in self-play mode, but this would
        // require too much end-to-end testing for our current scope (i.e. from scratch with fresh data each time),
        // so err on the proven side.
        return (_position.key() ^ ((_position.state_info()->repetition != 0) ? PredictionCache_IsRepetition : 0)) ^
            (_position.rule50_count() >= Config::Network.SelfPlay.TranspositionProgressThreshold
                ? PredictionCache_NoProgressCount[std::min(NoProgressSaturationCount, _position.rule50_count())] : 0);
    }
    else
    {
        // Stockfish key includes material and castling rights, so we just need to add history and no-progress.
        // Because Zobrist keys are iteratively updated via moves, they collide when combining positions that are
        // reached by different permutations of the same set of moves. So, rotate the key to differentiate.

        // Add no-progress plane to key.
        Key key = PredictionCache_NoProgressCount[std::min(NoProgressSaturationCount, _position.rule50_count())];

        // Add last 7 positions + 1 current position.
        // If there are fewer, no need to synthesize/saturate, since we're just differentating
        // positions/histories, not feeding planes into a neural network, so just stop rotating/hashing.
        const int historyPositionCount = std::min(INetwork::InputPreviousPositionCount + 0, static_cast<int>(_moves.size()));
        HistoryWalker<INetwork::InputPreviousPositionCount> history(this, historyPositionCount);
        int rotation = 0;
        while (history.Next())
        {
            key ^= Rotate(_position.key() ^ ((_position.state_info()->repetition != 0) ? PredictionCache_IsRepetition : 0), rotation++);
        }

        return key;
    }
}

void Game::GenerateImage(INetwork::InputPlanes& imageOut)
{
    GenerateImage(imageOut.data());
}

void Game::GenerateImage(INetwork::PackedPlane* imageOut)
{
    int nextPlane = 0;

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add last 7 positions' pieces and repetitions + 1 current position's pieces and repetitions, planes 0-103.
    // If any history positions are missing, saturate at the earliest history/current position.
    assert(nextPlane == 0);
    HistoryWalker<INetwork::InputPreviousPositionCount> history(this, INetwork::InputPreviousPositionCount);
    while (history.Next())
    {
        GeneratePieceAndRepetitionPlanes(imageOut, nextPlane, _position, history.Perspective());
        nextPlane += INetwork::InputPieceAndRepetitionPlanesPerPosition;
    }

    // Castling planes 104-107
    assert(nextPlane == 104);
    FillPlane(imageOut[nextPlane++], _position.can_castle(toPlay & KING_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(toPlay & QUEEN_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(~toPlay & KING_SIDE));
    FillPlane(imageOut[nextPlane++], _position.can_castle(~toPlay & QUEEN_SIDE));

    // No-progress plane 108
    // This is a special case, not to be bit-unpacked, but instead interpreted as an integer
    // to be normalized via the no-progress saturation count (99).
    assert(nextPlane == 108);
    imageOut[nextPlane++] = _position.rule50_count();

    static_assert(INetwork::InputPreviousPositionCount == 7);
    static_assert(INetwork::InputPieceAndRepetitionPlanesPerPosition == 13);
    static_assert(INetwork::InputAuxiliaryPlaneCount == 5);
    static_assert(INetwork::InputPlaneCount == 109);
    assert(nextPlane == INetwork::InputPlaneCount);
}

void Game::GenerateImageCompressed(INetwork::PackedPlane* piecesOut, INetwork::PackedPlane* auxiliaryOut) const
{
    int nextPieces = 0;

    // If it's black to play, flip the board and flip colors: always from the "current player's" perspective.
    const Color toPlay = ToPlay();

    // Add current position's pieces and repetitions, planes 91-103 (becoming future positions' planes 78-90, 65-77, etc.).
    GeneratePieceAndRepetitionPlanes(piecesOut, nextPieces, _position, toPlay);
    nextPieces += INetwork::InputPieceAndRepetitionPlanesPerPosition;

    // Castling planes 104-107
    int nextAuxiliary = 0;
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(toPlay & KING_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(toPlay & QUEEN_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(~toPlay & KING_SIDE));
    FillPlane(auxiliaryOut[nextAuxiliary++], _position.can_castle(~toPlay & QUEEN_SIDE));

    // No-progress plane 108
    // This is a special case, not to be bit-unpacked, but instead interpreted as an integer
    // to be normalized via the no-progress saturation count (99).
    auxiliaryOut[nextAuxiliary++] = _position.rule50_count();

    assert(nextPieces == INetwork::InputPieceAndRepetitionPlanesPerPosition);
    assert(nextAuxiliary == INetwork::InputAuxiliaryPlaneCount);
}

void Game::GenerateCommentaryImage(INetwork::CommentaryInputPlanes& imageOut)
{
    GenerateCommentaryImage(imageOut.data());
}

void Game::GenerateCommentaryImage(INetwork::PackedPlane* imageOut)
{
    // We need to provide positions before and after the move to comment on. This seems strange at first
    // because the image for a position includes history planes, but we provide the *output* of the
    // chess-playing model to the commentary decoder, not the *input*. Because of this, history information
    // may be consumed/lost. As an example, to decide whether a move was a blunder, we need to estimate the chance to win
    // at the previous and current positions, but the chess-playing model only returns one value for the current position,
    // at the head of the model.
    //
    // This is tricky: we need to nest HistoryWalker use, since "GenerateImage" uses one internally.
    // - HistoryWalker refers to Game::_position->current_state() rather than Game::_currentState, which makes this work.
    // - We just need to pop the latest off Game::_moves when using the outer HistoryWalker, since the inner HistoryWalker will refer to it.
    HistoryWalker<INetwork::InputPreviousPositionCount> previousPositionOuterWalker(this, 1);

    // Write a full image for the previous position.
    previousPositionOuterWalker.Next(); // Expect "true".
    Move latestMove = MOVE_NONE;
    if (!_moves.empty())
    {
        latestMove = _moves.back();
        _moves.pop_back();
    }
    GenerateImage(imageOut + 0);
    if (latestMove != MOVE_NONE)
    {
        _moves.push_back(latestMove);
    }

    // Write a full image for the current position.
    previousPositionOuterWalker.Next(); // Expect "true".
    GenerateImage(imageOut + INetwork::InputPlaneCount);
    previousPositionOuterWalker.Next(); // Expect "false".

    // Write a plane representing the side to move. The chess-playing model doesn't need this, but the commentary decoder does
    // in order to differentiate e.g. "white" vs. "black" and "e7" vs. "e2".
    FillPlane(imageOut[2 * INetwork::InputPlaneCount], ToPlay() == BLACK);

    static_assert(INetwork::CommentaryInputPlaneCount == 2 * INetwork::InputPlaneCount + 1);
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
        plane = QueenKnightPlane[Delta88(from, to)];
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

Position& Game::GetPosition()
{
    return _position;
}

const std::vector<Move>& Game::Moves() const
{
    return _moves;
}

std::vector<Move>& Game::Moves()
{
    return _moves;
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

void Game::GeneratePieceAndRepetitionPlanes(INetwork::PackedPlane* imageOut, int planeOffset, const Position& position, Color perspective) const
{
    // If it's black perspective, flip the board and flip colors.
    imageOut[planeOffset + 0] = position.pieces(perspective, PAWN);
    imageOut[planeOffset + 1] = position.pieces(perspective, KNIGHT);
    imageOut[planeOffset + 2] = position.pieces(perspective, BISHOP);
    imageOut[planeOffset + 3] = position.pieces(perspective, ROOK);
    imageOut[planeOffset + 4] = position.pieces(perspective, QUEEN);
    imageOut[planeOffset + 5] = position.pieces(perspective, KING);

    imageOut[planeOffset + 6] = position.pieces(~perspective, PAWN);
    imageOut[planeOffset + 7] = position.pieces(~perspective, KNIGHT);
    imageOut[planeOffset + 8] = position.pieces(~perspective, BISHOP);
    imageOut[planeOffset + 9] = position.pieces(~perspective, ROOK);
    imageOut[planeOffset + 10] = position.pieces(~perspective, QUEEN);
    imageOut[planeOffset + 11] = position.pieces(~perspective, KING);

    // No need to flip repetitions. Encode first time (1-repetition) as all zeros, second time (2-repetition) as all ones.
    // There should be no need to encode a 3-repetition because we don't predict for terminals (represented as negative "StateInfo::repetition").
    assert(position.state_info()->repetition >= 0);
    FillPlane(imageOut[planeOffset + 12], (position.state_info()->repetition != 0));

    static_assert(INetwork::InputPieceAndRepetitionPlanesPerPosition == 13);

    // Piece colors are already conditionally flipped via perspective/~perspective ordering. Flip all vertically if BLACK to play.
    if (perspective == BLACK)
    {
        // Only iterate over the piece planes: no need to flip repetitions.
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

// Operates over the Game's low-level Position, so the two will diverge until the final Next() is called.
template <int MaxHistoryMoves>
HistoryWalker<MaxHistoryMoves>::HistoryWalker(Game* game, int historyMoves)
    : _game(game)
    , _perspective(game->ToPlay())
    , _redoMoves{}
    , _redoStates{}
    , _redoCount(0)
    , _redoIndex(0)
{
    assert(historyMoves <= MaxHistoryMoves);

    for (int i = 0; i < historyMoves; i++)
    {
        // Leave room at index 0 for a synthetic MOVE_NONE for the initial Next() call.
        const int redoMoveIndex = (historyMoves - i);
        const int historyMoveIndex = (static_cast<int>(game->_moves.size()) - i - 1);
        if (historyMoveIndex >= 0)
        {
            // This is a real history move to undo. It may be a null move; e.g., for commentary games.
            const Move move = game->_moves[historyMoveIndex];
            _redoStates[redoMoveIndex] = game->_position.state_info();
            if (move != MOVE_NULL)
            {
                _game->_position.undo_move(move);
            }
            else
            {
                _game->_position.undo_null_move();
            }
            _redoMoves[redoMoveIndex] = move;
        }
        else
        {
            // There are no more history moves/positions, so saturate at the earliest history/current position.
            _redoMoves[redoMoveIndex] = MOVE_NONE;
        }
        _redoCount++;
        _perspective = ~_perspective;
    }

    // The initial Next() call needs to get us to the earliest history position/perspective, so use a synthetic MOVE_NONE at index 0.
    _redoMoves[0] = MOVE_NONE;
    _redoCount++;
    _perspective = ~_perspective;
}

template <int MaxHistoryMoves>
bool HistoryWalker<MaxHistoryMoves>::Next()
{
    // If e.g. historyMoves is 7 then Next() will return true 8 times, for 7 history positions plus 1 current position.
    if (_redoIndex >= _redoCount)
    {
        return false;
    }

    // Reapply the move if it exists. Otherwise, this was saturated from the earliest history/current position.
    const Move move = _redoMoves[_redoIndex];
    if (move != MOVE_NONE)
    {
        if (move != MOVE_NULL)
        {
            _game->_position.redo_move(move, *_redoStates[_redoIndex]);
        }
        else
        {
            _game->_position.redo_null_move(*_redoStates[_redoIndex]);
        }
    }

    // The perspective flips every history position, even for saturated positions, to give the neural network consistent inputs.
    // This doesn't matter for the starting position of a game, but does for set-up FEN positions (e.g. in STS strength tests).
    _perspective = ~_perspective;

    _redoIndex++;
    return true;
}

template <int MaxHistoryMoves>
Color HistoryWalker<MaxHistoryMoves>::Perspective()
{
    return _perspective;
}