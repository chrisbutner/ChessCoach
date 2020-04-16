#ifndef _SAVEDGAME_H_
#define _SAVEDGAME_H_

#include <vector>
#include <map>

#include <Stockfish/types.h>

struct SavedGame
{
public:

    SavedGame();
    SavedGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::map<Move, float>>& setChildVisits);
    SavedGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<std::map<Move, float>>&& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<std::map<Move, float>> childVisits;
};

#endif // _SAVEDGAME_H_