#ifndef _SAVEDGAME_H_
#define _SAVEDGAME_H_

#include <vector>
#include <map>
#include <set>

#include <Stockfish/types.h>

#include "Network.h"

struct SavedGame
{
    SavedGame();
    SavedGame(float setResult, const std::vector<Move>& setMoves, const std::vector<float>& setMctsValues, const std::vector<std::map<Move, float>>& setChildVisits);
    SavedGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<float>&& setMctsValues, std::vector<std::map<Move, float>>&& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<float> mctsValues;
    std::vector<std::map<Move, float>> childVisits;
};

struct SavedComment
{
    int moveIndex;
    std::vector<uint16_t> variationMoves;
    std::string comment;
};

struct SavedCommentary
{
    std::vector<SavedComment> comments;
};

struct Vocabulary
{
    int commentCount;
    std::set<std::string> vocabulary;
};

#endif // _SAVEDGAME_H_