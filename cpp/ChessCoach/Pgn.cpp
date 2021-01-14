#include "Pgn.h"

#include <string>
#include <sstream>
#include <limits>

#include <Stockfish/thread.h>
#include <Stockfish/movegen.h>

#include "Game.h"
#include "Preprocessing.h"

// Parsing is quite minimal and brittle, intended to parse very-well-formed PGNs with minimal dev time and improved only as needed.
// E.g. assume that pawn capture squares are fully specified.
//
// Note that this means SAN input like "ex" will cause overflows. Don't feed in untrusted input.
void Pgn::ParsePgn(std::istream& content, std::function<void(SavedGame&&, SavedCommentary&&)> gameHandler)
{
    while (true)
    {
        // Try to get the result from headers and validate against the one after the move list,
        // but be forgiving and allow the one after the move list to fill in.
        bool fenGame = false;
        float result = ParseHeaders(content, fenGame);

        // FEN games aren't supported, so skip.
        if (fenGame)
        {
            SkipGame(content);
            continue;
        }

        StateListPtr positionStates(new std::deque<StateInfo>(1));
        Position position;
        position.set(Config::StartingPosition, false /* isChess960 */, &positionStates->back(), Threads.main());
        std::vector<uint16_t> moves;
        SavedCommentary commentary;
        if (!ParseMoves(content, positionStates, position, moves, commentary, result, false /* inVariation */))
        {
            break;
        }
        if (std::isnan(result))
        {
            // This can be the end of the file, or a really bad PGN/game with no result in either headers or after the move list.
            break;
        }

        gameHandler(SavedGame(result, std::move(moves), GenerateMctsValues(moves, result), GenerateChildVisits(moves)), std::move(commentary));
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

float Pgn::ParseHeaders(std::istream& content, bool& fenGameInOut)
{
    float result = std::numeric_limits<float>::quiet_NaN();

    char c;
    while (content >> c)
    {
        if (c == '[')
        {
            ParseHeader(content, fenGameInOut, result);
        }
        else if (c == '1')
        {
            // Found the first move, so headers are finished.
            content.unget();
            break;
        }
        else if (c == '{')
        {
            // Found a comment before the first move, just throw it away.
            ParseUntil(content, '}');
        }
        else if ((c == 'ï') || (c == '»') || (c == '¿'))
        {
            // Ignore UTF-8 BOM.
        }
        else
        {
            // Let the move parser handle the rest.
            content.unget();
            break;
        }
    }

    return result;
}

void Pgn::ParseHeader(std::istream& content, bool& fenGameInOut, float& resultOut)
{
    const std::string fenHeader = "FEN";
    const std::string resultHeader = "Result";

    const int fenValueOffset = 5;
    const int resultValueOffset = 8;
    const std::string win = "1-";
    const std::string draw = "1/";
    const std::string undetermined = "*";
    const std::string loss = "0";

    // Grab the rest of the line. It would be nice to ParseUntil the closing square brace
    // but some PGNs like to nest square braces within headers like Event or Annotator.
    std::string header;
    std::getline(content, header);

    // FEN games aren't supported.
    if (header.compare(0, fenHeader.size(), fenHeader) == 0)
    {
        // Some PGNs include the starting position as a FEN header, which is pretty confusing.
        if (header.compare(fenValueOffset, std::size(Config::StartingPosition) - 1, Config::StartingPosition) != 0)
        {
            fenGameInOut = true;
        }
    }

    // Check for the "Result" header in particular and return the float representation.
    if (header.compare(0, resultHeader.size(), resultHeader) == 0)
    {
        if (header.compare(resultValueOffset, win.size(), win) == 0)
        {
            resultOut = CHESSCOACH_VALUE_WIN;
        }
        else if ((header.compare(resultValueOffset, draw.size(), draw) == 0) ||
            (header.compare(resultValueOffset, undetermined.size(), undetermined) == 0))
        {
            resultOut = CHESSCOACH_VALUE_DRAW;
        }
        else if (header.compare(resultValueOffset, loss.size(), loss) == 0)
        {
            resultOut = CHESSCOACH_VALUE_LOSS;
        }
    }
}

bool Pgn::ParseMoves(std::istream& content, StateListPtr& positionStates, Position& position, std::vector<uint16_t>& moves, SavedCommentary& commentary, float& resultInOut, bool inVariation)
{
    char c;
    std::string str;
    while (content >> c)
    {
        // Uncomment to debug: you can insert ^ in most places in the PGN.
        //if (c == '^')
        //{
        //    std::cout << position.fen() << std::endl;
        //    ::__debugbreak();
        //} else
        if (::isdigit(static_cast<unsigned char>(c)))
        {
            content.unget();
            int moveNumber;
            content >> moveNumber;

            if (((moveNumber == 1) && (moves.size() <= 1) && (content.peek() != '-') && (content.peek() != '/')) ||
                (moveNumber >= 2))
            {
                // After e.g. 8 half-moves, expect "5.": (moves.size() == (moveNumber - 1) * 2)
                // After e.g. 9 half-moves, expect "5...": (moves.size() == ((moveNumber - 1) * 2) + 1)
                //
                // However, poor-quality PGNs may do many things wrongly:
                // - Use the wrong move number
                // - Omit dots completely
                // - Include too few dots
                // - Abut the next move
                //
                // As long as the move number is unambiguous with results/castling, be forgiving.
                // Don't even try to consume the right number of dots, let dot-forgiveness handle it below.
                //
                // Do watch out for a 1-0 or 1/2-1/2 result without anything else in the move list though.
            }
            else if (moveNumber == 0)
            {
                // This is a loss ("0-1") or a poor-quality PGN with 0-0 or 0-0-0 instead
                // of O-O or O-O-O.
                //
                // Don't use another unget() here. It can fail on Windows on boundaries like 512000
                // despite reading the zero again in between ungets. ParseMoveGlyph will return an empty
                // string anyway if it hits whitespace right after the zero.
                ParseMoveGlyph(content, str);
                if (str == "-1")
                {
                    assert(!inVariation);
                    EncounterResult(CHESSCOACH_VALUE_LOSS, resultInOut);

                    // Game is finished.
                    break;
                }
                else
                {
                    std::replace(str.begin(), str.end(), '0', 'O');
                    str.insert(str.begin(), 'O');
                    const Move move = ParseSan(position, str);
                    if (move == MOVE_NONE)
                    {
                        assert(!"Unexpected zero in PGN, not 0-1 or 0-0 or 0-0-0");
                        return false;
                    }
                    ApplyMove(positionStates, position, move);
                    moves.push_back(static_cast<uint16_t>(move));
                }
            }
            else if (moveNumber == 1)
            {
                // This is a win ("1-0") or draw ("1/2-1/2").
                content >> c; // Could fail and remain '1'
                if (c == '-')
                {
                    assert(!inVariation);
                    if (!Expect(content, "0"))
                    {
                        return false;
                    }
                    EncounterResult(CHESSCOACH_VALUE_WIN, resultInOut);

                    // Game is finished.
                    break;
                }
                else if (c == '/')
                {
                    // Also be forgiving of just "1/2".
                    assert(!inVariation);
                    ParseMoveGlyph(content, str);
                    if ((str == "2-1/2") || (str == "2"))
                    {
                        EncounterResult(CHESSCOACH_VALUE_DRAW, resultInOut);

                        // Game is finished.
                        break;
                    }
                    else
                    {
                        assert(!"Unexpected 1/ in PGN, not 1/2-1/2 or 1/2");
                        return false;
                    }
                }
                else
                {
                    assert(!"Unexpected move number in PGN");
                    return false;
                }
            }
            else
            {
                assert(!"Unexpected move number in PGN");
                return false;
            }
        }
        else if (c == '.')
        {
            // Allow for too few dots in the move number cases above, as well as too many dots
            // in cases like  "38. ...Kf3". Just consume and move on.
        }
        else if (c == '*')
        {
            // This is an indeterminate result: "*".
            assert(!inVariation);
            EncounterResult(CHESSCOACH_VALUE_DRAW, resultInOut);

            // Game is finished.
            break;
        }
        // Allow "--" or "Z0" for null moves.
        else if (::isalpha(static_cast<unsigned char>(c)) || (c == '-'))
        {
            content.unget();
            ParseMoveGlyph(content, str);
            const Move move = ParseSan(position, str);
            if (move == MOVE_NONE)
            {
                assert(!"Failed to parse move SAN in PGN");
                return false;
            }
            ApplyMove(positionStates, position, move);
            moves.push_back(static_cast<uint16_t>(move));
        }
        else if (c == '$')
        {
            // Ignore the rest of the Numeric Annotation Glyph (NAG).
            ParseMoveGlyph(content, str);
        }
        else if ((c == '!') || (c == '?'))
        {
            // Ignore isolated unencoded NAGs.
        }
        else if (c == '{')
        {
            ParseComment(content, moves, commentary);
        }
        else if (c == '(')
        {
            if (!ParseVariation(content, position, moves, commentary, resultInOut))
            {
                return false;
            }
        }
        else if (c == ')')
        {
            if (!inVariation)
            {
                assert(!"Unexpected ) outside of a variation");
                return false;
            }

            // Variation is finished.
            break;
        }
        else if (c == '[')
        {
            // Assume we're in a poor-quality PGN with a missing result after the move list.
            // Try to fill in a result then move on to the next game.
            if (inVariation)
            {
                assert(!"Unexpected [ inside a variation");
                return false;
            }

            float fillResult = resultInOut;
            if (std::isnan(fillResult))
            {
                fillResult = CHESSCOACH_VALUE_DRAW;
            }
            
            EncounterResult(fillResult, resultInOut);
            content.unget();

            // Game is finished.
            break;
        }
        else if (c == '}')
        {
            // Assume a poor-quality PGN with something like "{ comment } }".
            // Just consume and move on.
        }
        else
        {
            assert(!"Unexpected character in PGN");
            return false;
        }
    }

    return true;
}

void Pgn::ParseComment(std::istream& content, const std::vector<uint16_t>& moves, SavedCommentary& commentary)
{
    // Grab the full comment string between curly braces, including spaces.
    std::string comment = ParseUntil(content, '}');

    // Store the trimmed comment if not empty.
    Preprocessor::Trim(comment);
    if (!comment.empty())
    {
        commentary.comments.emplace_back(SavedComment{ (static_cast<int>(moves.size()) - 1), {}, comment });
    }
}

std::string Pgn::ParseUntil(std::istream& content, char delimiter)
{
    // Grab the full string including spaces, and consume the delimiter.
    std::string text;
    char c;

    content >> std::noskipws;
    while ((content >> c) && (c != delimiter))
    {
        text += c;
    }
    content >> std::skipws;

    return text;
}

void Pgn::ParseMoveGlyph(std::istream& content, std::string& target)
{
    // Stop at the first whitespace or punctuation, leaving them unconsumed.
    target.clear();
    char c;

    content >> std::noskipws;
    while (content >> c)
    {
        if (::isspace(static_cast<unsigned char>(c)) ||
            (c == '(') || (c == ')') || (c == '{') || (c == '}'))
        {
            content.unget();
            break;
        }
        target += c;
    }
    content >> std::skipws;
}

bool Pgn::ParseVariation(std::istream& content, const Position& parent, const std::vector<uint16_t>& parentMoves, SavedCommentary& commentary, float& resultInOut)
{
    // Copy/branch the position and move list. We can start a new empty StateInfo list
    // that refers into the old one just like the "Game" class.
    StateListPtr positionStates(new std::deque<StateInfo>());
    Position position = parent;
    std::vector<uint16_t> moves = parentMoves;

    UndoMoveInVariation(position, Move(moves.back()));
    moves.pop_back();

    int newCommentary = static_cast<int>(commentary.comments.size());
    const bool success = ParseMoves(content, positionStates, position, moves, commentary, resultInOut, true /* inVariation */);

    // Loop over all new comments since they're specific to this variation.
    // This means that right now their moveIndex will correctly reference
    // this variation's "moves" but will dangle over "parentMoves", after
    // culling the final move that the variation is overriding.
    //
    // Note that since variations can nest, some comments may already have
    // "variationMoves", so these need to be preserved. This is how we know
    // that they've already been "fixed up" to lie within "moves" though.
    const int preVariationMoveCount = (static_cast<int>(parentMoves.size()) - 1);
    while (newCommentary < commentary.comments.size())
    {
        // One annoying special case is when annotators integrate moves into prose, e.g.:
        //
        // 11...Nxd5 ( { Another plan is } 11...Bxd5 12. exd5 { when Nf6 remains in place to threaten Pd5 }
        //
        // Here, the comment "Another plan is" helps refer to the next move, not the previous, but human input would be needed
        // to combine/rewrite in a post-move fashion, e.g. "Another plan, where Nf6 remains in place to threaten Pd5".
        // Sometimes you'd want to combine with the next comment, other times move the comment to the end of the variation, etc.
        // It seems like a very complicated problem after browsing through various annotators' work.
        //
        // For now, throw away the initial pre-move comment. When the comment was generated the final parent move
        // had already been culled from "moves", so we can test for (comment.moveIndex < preVariationMoveCount)
        // (i.e. breaking the first asserted invariant below).
        SavedComment& comment = commentary.comments[newCommentary];
        if (comment.moveIndex < preVariationMoveCount)
        {
            commentary.comments.erase(commentary.comments.begin() + newCommentary);
            continue;
        }
        
        assert(comment.moveIndex >= preVariationMoveCount);
        assert(comment.moveIndex < moves.size());

        // Pull the comment's moveIndex back until it can point into "parentMoves",
        // after culling the final move that the variation is overriding,
        // and prepend the backtracked moves into "variationMoves" (which may be
        // non-empty because of a nested variation's fix-up).
        const int target = (preVariationMoveCount - 1);
        const int depth = (comment.moveIndex - target);
        comment.variationMoves.insert(comment.variationMoves.begin(),
            moves.begin() + preVariationMoveCount, moves.begin() + preVariationMoveCount + depth);
        comment.moveIndex -= depth;
        assert(comment.moveIndex == target);

        newCommentary++;
    }

    return success;
}

bool Pgn::Expect(std::istream& content, char expected)
{
    char c;
    if (!(content >> c))
    {
        assert(!"Unexpected end-of-stream");
        return false;
    }

    if (c != expected)
    {
        assert(!"Unexpected char found");
        return false;
    }

    return true;
}

bool Pgn::Expect(std::istream& content, const std::string& expected)
{
    for (const char e : expected)
    {
        if (!Expect(content, e))
        {
            return false;
        }
    }

    return true;
}

void Pgn::EncounterResult(float encountered, float& resultInOut)
{
    // Some PGNs may not include the actual result in the header (e.g. "?" instead).
    // Be forgiving and let the result after the move list fill it in.
    if (!std::isnan(resultInOut))
    {
        assert(encountered == resultInOut);
    }
    else
    {
        resultInOut = encountered;
    }
}

void Pgn::SkipGame(std::istream& content)
{
    // We want to ParseUntil the start of the next header ('[') but sometimes
    // square brackets are used inside comments, so ignore them there. Don't
    // try to handle nested comments, since that would be very destructive
    // elsewhere anyway. It's fine to skip whitespace in this case.
    bool inComment = false;
    char c;
    while (content >> c)
    {
        if (c == '{')
        {
            inComment = true;
        }
        else if (c == '}')
        {
            inComment = false;
        }
        else if (c == '[')
        {
            if (!inComment)
            {
                content.unget();
                break;
            }
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
    if ((san == "--") || (san == "Z0"))
    {
        return MOVE_NULL;
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

std::string Pgn::San(const std::string& fen, Move move, bool showCheckmate)
{
    Position position;
    StateInfo state;
    position.set(fen, false /* isChess960 */, &state, Threads.main());
    return San(position, move, showCheckmate);
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
    if (move != MOVE_NULL)
    {
        position.do_move(move, positionStates->back());
    }
    else
    {
        position.do_null_move(positionStates->back());
    }
}

// Intended for branched variations, so doesn't update the StateInfo list. 
void Pgn::UndoMoveInVariation(Position& position, Move move)
{
    if (move != MOVE_NULL)
    {
        position.undo_move(move);
    }
    else
    {
        position.undo_null_move();
    }
}

#pragma warning(disable:4706) // Intentionally assigning, not comparing
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
#pragma warning(default:4706) // Intentionally assigning, not comparing

#pragma warning(disable:4706) // Intentionally assigning, not comparing
Move Pgn::ParsePawnSan(const Position& position, const std::string& san)
{
    const Color toPlay = position.side_to_move();

    // Handle e.g. "gf6<...>" or "g5f6<...>" as "gxf6<...>".
    size_t capture = san.find('x', 1);
    if (capture == std::string::npos)
    {
        File maybeFile;

        if ((san.size() >= 2) &&
            (maybeFile = ParseFile(san, 1)) && // Intentionally assigning, not comparing
            (maybeFile >= FILE_A) &&
            (maybeFile <= FILE_H))
        {
            capture = 0;
        }
        else if ((san.size() >= 3) &&
            (maybeFile = ParseFile(san, 2)) && // Intentionally assigning, not comparing
            (maybeFile >= FILE_A) &&
            (maybeFile <= FILE_H))
        {
            capture = 1;
        }
    }

    const Square targetSquare = ParseSquare(san, (capture != std::string::npos) ? (static_cast<int>(capture) + 1) : 0);
    const Direction advance = ((toPlay == WHITE) ? NORTH : SOUTH);
    Square fromSquare = (targetSquare - advance);
    const size_t promotion = san.find('=', 2);

    if (capture != std::string::npos)
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
    else if ((capture != std::string::npos) && (position.piece_on(targetSquare) == NO_PIECE))
    {
        return make<ENPASSANT>(fromSquare, targetSquare);
    }

    return make_move(fromSquare, targetSquare);
}
#pragma warning(default:4706) // Intentionally assigning, not comparing

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