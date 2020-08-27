#include "Pgn.h"

#include <string>
#include <sstream>
#include <limits>

#include <Stockfish/thread.h>
#include <Stockfish/movegen.h>

#include "Game.h"

// Parsing is quite minimal and brittle, intended to parse very-well-formed PGNs with minimal dev time and improved only as needed.
// E.g. assume that pawn capture squares are fully specified
void Pgn::ParsePgn(std::istream& content, std::function<void(SavedGame&&)> gameHandler)
{
    while (true)
    {
        float result = ParseResult(content);
        if (std::isnan(result))
        {
            break;
        }

        std::vector<uint16_t> moves;
        ParseMoves(content, moves, result);

        gameHandler(SavedGame(result, std::move(moves), GenerateMctsValues(moves, result), GenerateChildVisits(moves)));
    }
}

std::vector<float> Pgn::GenerateMctsValues(const std::vector<uint16_t>& moves, float result)
{
    std::vector<float> mctsValues(moves.size());

    // There is no MCTS search data, so set the MCTS value to the original game's result
    // Flip to the side to play's perspective (NOT the parent's perspective, like in MCTS trees).
    for (int i = 0; i < moves.size(); i++)
    {
        const Color toPlay = Color(i % COLOR_NB);
        mctsValues[i] = Game::FlipValue(toPlay, result);
    }

    return mctsValues;
}

std::vector<std::map<Move, float>> Pgn::GenerateChildVisits(const std::vector<uint16_t>& moves)
{
    std::vector<std::map<Move, float>> childVisits(moves.size());

    // There is no MCTS search data, so just set 1.0 for the chosen move and imply 0.0 for others.
    for (int i = 0; i < moves.size(); i++)
    {
        childVisits[i].emplace(Move(moves[i]), 1.f);
    }

    return childVisits;
}

// Assume BOM + [Result on the same line won't happen.
float Pgn::ParseResult(std::istream& content) 
{
    const std::string resultHeader = "[Result";

    const int resultValueOffset = 9;
    const std::string win = "1-";
    const std::string draw = "1/";
    const std::string undetermined = "*";
    const std::string loss = "0";

    std::string line;
    while (std::getline(content, line))
    {
        if (line.compare(0, resultHeader.size(), resultHeader) == 0)
        {
            if (line.compare(resultValueOffset, win.size(), win) == 0)
            {
                return CHESSCOACH_VALUE_WIN;
            }
            else if ((line.compare(resultValueOffset, draw.size(), draw) == 0) ||
                (line.compare(resultValueOffset, undetermined.size(), undetermined) == 0))
            {
                return CHESSCOACH_VALUE_DRAW;
            }
            else if (line.compare(resultValueOffset, loss.size(), loss) == 0)
            {
                return CHESSCOACH_VALUE_LOSS;
            }
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}

void Pgn::ParseMoves(std::istream& content, std::vector<uint16_t>& moves, float result)
{
    const std::string firstMove = "1";

    std::string line;
    while (std::getline(content, line))
    {
        if (line.compare(0, firstMove.size(), firstMove) != 0)
        {
            continue;
        }

        std::stringstream moveStream(line);
        std::string extra;

        StateListPtr positionStates(new std::deque<StateInfo>(1));
        Position position;
        position.set(Config::StartingPosition, false /* isChess960 */, &positionStates->back(), Threads.main());

        moveStream >> extra;

        while (true)
        {
            std::string san1;
            std::string san2;

            moveStream >> san1 >> san2 >> extra;
            
            Move move1 = ParseSan(position, san1);
            if (move1 == MOVE_NONE)
            {
                // E.g. finishes with "66. Kf4 Be5+ 0-1" (this is the previous "extra").
                assert(ParseResultPrecise(extra) == result);
                return;
            }
            ApplyMove(positionStates, position, move1);
            moves.push_back(move1);

            Move move2 = ParseSan(position, san2);
            if (move2 == MOVE_NONE)
            {
                // E.g. finishes with "68. Kb1 1/2-1/2".
                assert(ParseResultPrecise(san2) == result);
                return;
            }
            ApplyMove(positionStates, position, move2);
            moves.push_back(move2);
        }
    }
}

// Returns MOVE_NONE for failure.
Move Pgn::ParseSan(const Position& position, const std::string& san)
{
    const std::string queenside = "O-O-O";
    
    const Color toPlay = position.side_to_move();

    if (san.empty())
    {
        return MOVE_NONE;
    }

    // E.g. finishes with "68. Kb1 1/2-1/2".
    if ((san[0] == '1') || (san[0] == '0') || (san[0] == '*'))
    {
        return MOVE_NONE;
    }

    const PieceType fromPieceType = ParsePieceType(san, 0);
    switch (fromPieceType)
    {
    case NO_PIECE_TYPE:
        if (san.compare(0, queenside.size(), queenside) == 0)
        {
            return make<CASTLING>(position.square<KING>(toPlay), position.castling_rook_square(toPlay & QUEEN_SIDE));
        }
        return make<CASTLING>(position.square<KING>(toPlay), position.castling_rook_square(toPlay & KING_SIDE));
    case KING:
        return make_move(position.square<KING>(toPlay), (san[1] == 'x') ? ParseSquare(san, 2) : ParseSquare(san, 1));
    case PAWN:
        return ParsePawnSan(position, san);
    default:
        return ParsePieceSan(position, san, fromPieceType);
    }
}

// Always writes 1/2-1/2 rather than * in the context of ChessCoach, where games are technically
// adjudicated as drawn at 512 moves rather than undetermined.
void Pgn::GeneratePgn(std::ostream& content, const SavedGame& game)
{
    std::string result;
    if (game.result == CHESSCOACH_VALUE_WIN)
    {
        result = "1-0";
    }
    else if (game.result == CHESSCOACH_VALUE_LOSS)
    {
        result = "0-1";
    }
    else
    {
        result = "1/2-1/2";
    }

    content << "[Event \"\"]" << std::endl
        << "[Site \"\"]" << std::endl
        << "[Date \"\"]" << std::endl
        << "[Round \"\"]" << std::endl
        << "[White \"\"]" << std::endl
        << "[Black \"\"]" << std::endl
        << "[Result \"" << result << "\"]" << std::endl
        << std::endl;

    StateListPtr positionStates(new std::deque<StateInfo>(1));
    Position position;
    position.set(Config::StartingPosition, false /* isChess960 */, &positionStates->back(), Threads.main());

    for (int i = 0; i < game.moveCount; i++)
    {
        if ((i % 2) == 0)
        {
            content << ((i / 2) + 1) << ". ";
        }

        const Move move = Move(game.moves[i]);
        content << San(position, move, (i == (game.moveCount - 1))) << " ";
        ApplyMove(positionStates, position, move);
    }

    content << result << std::endl
        << std::endl;
}

std::string Pgn::San(const Position& position, Move move, bool showCheckmate)
{
    if (move == MOVE_NONE)
    {
        return "none";
    }
    if (move == MOVE_NULL)
    {
        return "null";
    }

    std::string san;
    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const Piece piece = position.piece_on(from);
    const PieceType pieceType = type_of(piece);

    // Castling/pawn/piece
    if (type_of(move) == CASTLING)
    {
        const bool kingside = (to > from);
        san = (kingside ? "O-O" : "O-O-O");
    }
    else if (pieceType == PAWN)
    {
        san = SanPawn(position, move);
    }
    else
    {
        san = SanPiece(position, move, piece);
    }

    // Suffix
    if (position.gives_check(move))
    {
        bool checkmate = false;

        if (showCheckmate)
        {
            StateInfo state;
            Position& mutablePosition = const_cast<Position&>(position);
            mutablePosition.do_move(move, state, true /* givesCheck */);
            checkmate = (MoveList<LEGAL>(mutablePosition).size() == 0);
            mutablePosition.undo_move(move);
        }

        if (checkmate)
        {
            san += '#';
        }
        else
        {
            san += '+';
        }
    }

    return san;
}

std::string Pgn::SanPawn(const Position& position, Move move)
{
    std::string san;
    san.reserve(8);

    const Square from = from_sq(move);
    const Square to = to_sq(move);

    // Capture
    const bool capture = ((type_of(move) == ENPASSANT) || (position.piece_on(to) != NO_PIECE));
    if (capture)
    {
        san += FileSymbol(file_of(from));
        san += 'x';
    }

    // Target square
    san += FileSymbol(file_of(to));
    san += RankSymbol(rank_of(to));

    // Promotion
    if (type_of(move) == PROMOTION)
    {
        san += '=';
        san += PieceSymbol[promotion_type(move)];
    }

    return san;
}

std::string Pgn::SanPiece(const Position& position, Move move, Piece piece)
{
    std::string san;
    san.reserve(8);

    const Square from = from_sq(move);
    const Square to = to_sq(move);
    const PieceType pieceType = type_of(piece);

    // Piece type
    san += PieceSymbol[pieceType];

    const Bitboard fromPieces = Attacks(position, type_of(piece), to);
    assert(fromPieces);

    // Disambiguation
    const Bitboard otherPieces = (fromPieces & ~square_bb(from));
    const Bitboard otherPiecesLegal = Legal(position, otherPieces, to);
    if (otherPiecesLegal)
    {
        const File fromFile = file_of(from);
        const Rank fromRank = rank_of(from);
        const Bitboard fromFileMask = file_bb(fromFile);
        const Bitboard fromRankMask = rank_bb(fromRank);
        
        // Prefer file->rank->both.
        if (!(otherPiecesLegal & fromFileMask))
        {
            san += FileSymbol(fromFile);
        }
        else if (!(otherPiecesLegal & fromRankMask))
        {
            san += RankSymbol(fromRank);
        }
        else
        {
            // Tautology with "& ~square_bb(from)" but check variable flow.
            assert(!(otherPiecesLegal & fromFileMask & fromRankMask));
            san += FileSymbol(fromFile);
            san += RankSymbol(fromRank);
        }
    }

    // Capture
    if (position.piece_on(to) != NO_PIECE)
    {
        san += 'x';
    }

    // Target square
    san += FileSymbol(file_of(to));
    san += RankSymbol(rank_of(to));

    return san;
}

void Pgn::ApplyMove(StateListPtr& positionStates, Position& position, Move move)
{
    positionStates->emplace_back();
    position.do_move(move, positionStates->back());
}

float Pgn::ParseResultPrecise(const std::string& text)
{
    if (text == "1-0")
    {
        return CHESSCOACH_VALUE_WIN;
    }
    else if (text == "1/2-1/2")
    {
        return CHESSCOACH_VALUE_DRAW;
    }
    else if (text == "*")
    {
        return CHESSCOACH_VALUE_DRAW;
    }
    else if (text == "0-1")
    {
        return CHESSCOACH_VALUE_LOSS;
    }
    else
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

Move Pgn::ParsePieceSan(const Position& position, const std::string& san, PieceType fromPieceType)
{
    const size_t capture = san.find('x', 1);
    Square targetSquare;
    bool hasPartialDisambiguation = false;
    bool hasFullDisambiguation = false;

    if (capture != std::string::npos)
    {
        targetSquare = ParseSquare(san, static_cast<int>(capture) + 1);
        hasFullDisambiguation = (capture >= 3);
        hasPartialDisambiguation = (!hasFullDisambiguation && (capture >= 2));
    }
    else
    {
        Rank maybeRank = ParseRank(san, 2);
        if ((maybeRank >= RANK_1) && (maybeRank <= RANK_8))
        {
            // Check for full disambiguation.
            if ((san.size() >= 5) &&
                (maybeRank = ParseRank(san, 4)) && // Intentionally assigning, not comparing
                (maybeRank >= RANK_1) &&
                (maybeRank <= RANK_8))
            {
                targetSquare = ParseSquare(san, 3);
                hasFullDisambiguation = true;
            }
            else
            {
                targetSquare = ParseSquare(san, 1);
            }
        }
        else
        {
            targetSquare = ParseSquare(san, 2);
            hasPartialDisambiguation = true;
        }
    }

    Bitboard fromPieces = Attacks(position, fromPieceType, targetSquare);
    assert(fromPieces);
    
    if (more_than_one(fromPieces))
    {
        if (hasFullDisambiguation)
        {
            Square disambiguationSquare = ParseSquare(san, 1);
            fromPieces &= square_bb(disambiguationSquare);
        }
        else if (hasPartialDisambiguation)
        {
            // E.g. "Nfd7"
            File disambiguationFile = ParseFile(san, 1);
            if ((disambiguationFile >= FILE_A) && (disambiguationFile <= FILE_H))
            {
                fromPieces &= file_bb(disambiguationFile);
            }
            // E.g. "R6h2"
            else
            {
                Rank disambiguationRank = ParseRank(san, 1);
                assert((disambiguationRank >= RANK_1) && (disambiguationRank <= RANK_8));

                fromPieces &= rank_bb(disambiguationRank);
            }
        }
        else
        {
            // Easier example: knights on c3 (pinned), g1, move is Ne2
            // More difficult example: king on H8, rooks on E8 and G7, both pinned, move is Rg8.
            Move legal = MOVE_NONE;
            while (fromPieces)
            {
                Move candidate = make_move(pop_lsb(&fromPieces), targetSquare);
                if (position.legal(candidate))
                {
                    assert(legal == MOVE_NONE);
                    legal = candidate;
                }
            }
            return legal;
        }

#ifdef _DEBUG
        if (!fromPieces || more_than_one(fromPieces))
        {
            std::cout << position.fen() << std::endl;
            std::cout << san << std::endl;
        }
#endif
        assert(fromPieces);
        assert(!more_than_one(fromPieces));
    }

    return make_move(lsb(fromPieces), targetSquare);
}

Move Pgn::ParsePawnSan(const Position& position, const std::string& san)
{
    const Color toPlay = position.side_to_move();
    const bool capture = (san[1] == 'x');
    const Square targetSquare = ParseSquare(san, capture ? 2 : 0);
    const Direction advance = ((toPlay == WHITE) ? NORTH : SOUTH);
    Square fromSquare = (targetSquare - advance);
    const size_t promotion = san.find('=', 2);

    if (capture)
    {
        fromSquare = make_square(ParseFile(san, 0), rank_of(fromSquare));
    }
    else if (position.piece_on(fromSquare) != make_piece(toPlay, PAWN))
    {
        fromSquare -= advance;
    }

    if (promotion != std::string::npos)
    {
        PieceType promotionType = ParsePieceType(san, static_cast<int>(promotion) + 1);

        return make<PROMOTION>(fromSquare, targetSquare, promotionType);
    }
    else if (capture && (position.piece_on(targetSquare) == NO_PIECE))
    {
        return make<ENPASSANT>(fromSquare, targetSquare);
    }

    return make_move(fromSquare, targetSquare);
}

Square Pgn::ParseSquare(const std::string& text, int offset)
{
    return make_square(ParseFile(text, offset), ParseRank(text, offset + 1));
}

File Pgn::ParseFile(const std::string& text, int offset)
{
    return File(text[offset] - 'a');
}

Rank Pgn::ParseRank(const std::string& text, int offset)
{
    return Rank(text[offset] - '1');
}

// Returns NO_PIECE_TYPE for castling.
PieceType Pgn::ParsePieceType(const std::string& text, int offset)
{
    switch (text[offset])
    {
    case 'N': return KNIGHT;
    case 'B': return BISHOP;
    case 'R': return ROOK;
    case 'Q': return QUEEN;
    case 'K': return KING;
    case 'O': return NO_PIECE_TYPE;
    default:
        assert((text[offset] >= 'a') && (text[offset] <= 'h'));
        return PAWN;
    }
}

Bitboard Pgn::Attacks(const Position& position, PieceType pieceType, Square targetSquare)
{
    const Bitboard from = position.pieces(position.side_to_move(), pieceType);
    const Bitboard attackers = position.attacks_from(pieceType, targetSquare);
    const Bitboard fromPieces = (attackers & from);
    return fromPieces;
}

Bitboard Pgn::Legal(const Position& position, Bitboard fromPieces, Square targetSquare)
{
    Bitboard legal = fromPieces;
    while (fromPieces)
    {
        const Square from = pop_lsb(&fromPieces);
        const Move candidate = make_move(from, targetSquare);
        if (!position.legal(candidate))
        {
            legal ^= square_bb(from);
        }
    }
    return legal;
}