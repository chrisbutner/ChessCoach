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
    std::vector<std::string> vocabulary;
};

#endif // _SAVEDGAME_H_