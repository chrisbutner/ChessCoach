#ifndef _SAVEDGAME_H_
#define _SAVEDGAME_H_

#include <vector>
#include <map>

#include <Stockfish/types.h>

#include "Network.h"

struct SavedGame
{
public:

    SavedGame();
    SavedGame(float setResult, const std::vector<Move>& setMoves, const std::vector<float>& setMctsValues, const std::vector<std::map<Move, float>>& setChildVisits);
    SavedGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<float>&& setMctsValues, std::vector<std::map<Move, float>>&& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<float> mctsValues;
    std::vector<std::map<Move, float>> childVisits;
};

struct Comment
{
public:

    int moveIndex;
    std::vector<uint16_t> variationMoves;
    std::string comment;
};

struct Commentary
{
public:

    std::vector<Comment> comments;
};

struct SavedComment
{
public:

    SavedComment();
    SavedComment(int setGameIndex, int setMoveIndex, std::vector<uint16_t>&& setVariationMoves, std::string&& setComment);

    int gameIndex;
    int moveIndex;
    std::vector<uint16_t> variationMoves;
    std::string comment;
};

struct SavedCommentary
{
public:

    std::vector<SavedGame> games;
    std::vector<SavedComment> comments;
};

#endif // _SAVEDGAME_H_