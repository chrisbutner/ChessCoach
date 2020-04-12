#include "Pgn.h"

#include <string>
#include <sstream>
#include <limits>

#include <Stockfish/thread.h>
#include <Stockfish/movegen.h>

#include "Game.h"

// Parsing is quite minimal and brittle, intended to parse very-well-formed PGNs with minimal dev time and improved only as needed.
// E.g. assume that pawn capture squares are fully specified
// Assume UTF-8-ish encoding.
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

        gameHandler(SavedGame(result, std::move(moves), GenerateChildVisits(moves)));
    }
}

std::vector<std::map<Move, float>> Pgn::GenerateChildVisits(const std::vector<uint16_t>& moves)
{
    std::vector<std::map<Move, float>> childVisits(moves.size());

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
Move Pgn::ParseSan(Position& position, const std::string& san)
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

Move Pgn::ParsePieceSan(Position& position, const std::string& san, PieceType fromPieceType)
{
    const Color toPlay = position.side_to_move();
    const size_t capture = san.find('x', 1);
    Square targetSquare;
    bool hasDisambiguation;

    if (capture != std::string::npos)
    {
        targetSquare = ParseSquare(san, static_cast<int>(capture) + 1);
        hasDisambiguation = (capture >= 2);
    }
    else
    {
        Rank maybeRank = ParseRank(san, 2);
        if ((maybeRank >= RANK_1) && (maybeRank <= RANK_8))
        {
            targetSquare = ParseSquare(san, 1);
            hasDisambiguation = false;
        }
        else
        {
            targetSquare = ParseSquare(san, 2);
            hasDisambiguation = true;
        }
    }

    const Bitboard from = position.pieces(toPlay, fromPieceType);
    const Bitboard attackers = position.attacks_from(fromPieceType, targetSquare);
    Bitboard fromPieces = (attackers & from);
    assert(fromPieces);
    
    if (more_than_one(fromPieces))
    {
        if (!hasDisambiguation)
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
        else
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

        assert(fromPieces);
        assert(!more_than_one(fromPieces));
    }

    return make_move(lsb(fromPieces), targetSquare);
}

Move Pgn::ParsePawnSan(Position& position, const std::string& san)
{
    const Color toPlay = position.side_to_move();
    const bool capture = (san[1] == 'x');
    const Square targetSquare = ParseSquare(san, capture ? 2 : 0);
    const Direction up = ((toPlay == WHITE) ? NORTH : SOUTH);
    Square fromSquare = (targetSquare - up);
    const size_t promotion = san.find('=', 2);

    if (capture)
    {
        fromSquare = make_square(ParseFile(san, 0), rank_of(fromSquare));
    }
    else if (position.piece_on(fromSquare) != make_piece(toPlay, PAWN))
    {
        fromSquare -= up;
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