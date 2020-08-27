#ifndef _PGN_H_
#define _PGN_H_

#include <iostream>
#include <functional>

#include <Stockfish/position.h>

#include "SavedGame.h"

class Pgn
{
public:

    static void ParsePgn(std::istream& content, std::function<void(SavedGame&&)> gameHandler);
    static Move ParseSan(const Position& position, const std::string& san);

    static void GeneratePgn(std::ostream& content, const SavedGame& game);
    static std::string San(const Position& position, Move move, bool showCheckmate);

private:

    static constexpr const char PieceSymbol[PIECE_TYPE_NB] = { '-', '-', 'N', 'B', 'R', 'Q', 'K', '-' };

private:

    static std::vector<float> GenerateMctsValues(const std::vector<uint16_t>& moves, float result);
    static std::vector<std::map<Move, float>> GenerateChildVisits(const std::vector<uint16_t>& moves);
    static float ParseResult(std::istream& content);
    static void ParseMoves(std::istream& content, std::vector<uint16_t>& moves, float result);
    static void ApplyMove(StateListPtr& positionStates, Position& position, Move move);
    static float ParseResultPrecise(const std::string& text);
    static Move ParsePieceSan(const Position& position, const std::string& san, PieceType fromPieceType);
    static Move ParsePawnSan(const Position& position, const std::string& san);
    static Square ParseSquare(const std::string& text, int offset);
    static File ParseFile(const std::string& text, int offset);
    static Rank ParseRank(const std::string& text, int offset);
    static PieceType ParsePieceType(const std::string& text, int offset);

    static std::string SanPawn(const Position& position, Move move);
    static std::string SanPiece(const Position& position, Move move, Piece piece);

    static Bitboard Attacks(const Position& position, PieceType pieceType, Square targetSquare);
    static Bitboard Legal(const Position& position, Bitboard fromPieces, Square targetSquare);

    static inline char FileSymbol(File file) { return 'a' + file; }
    static inline char RankSymbol(Rank rank) { return '1' + rank; }
};

#endif // _PGN_H_