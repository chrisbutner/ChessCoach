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
    static Move ParseSan(Position& position, const std::string& san);

private:

    static std::vector<std::map<Move, float>> GenerateChildVisits(const std::vector<uint16_t>& moves);
    static float ParseResult(std::istream& content);
    static void ParseMoves(std::istream& content, std::vector<uint16_t>& moves, float result);
    static void ApplyMove(StateListPtr& positionStates, Position& position, Move move);
    static float ParseResultPrecise(const std::string& text);
    static Move ParsePieceSan(Position& position, const std::string& san, PieceType fromPieceType);
    static Move ParsePawnSan(Position& position, const std::string& san);
    static Square ParseSquare(const std::string& text, int offset);
    static File ParseFile(const std::string& text, int offset);
    static Rank ParseRank(const std::string& text, int offset);
    static PieceType ParsePieceType(const std::string& text, int offset);
};

#endif // _PGN_H_