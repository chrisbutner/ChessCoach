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

#include "SavedGame.h"

SavedGame::SavedGame()
    : result(-1.0f)
    , moveCount(0)
{
}

SavedGame::SavedGame(float setResult, const std::vector<Move>& setMoves, const std::vector<float>& setMctsValues, const std::vector<std::map<Move, float>>& setChildVisits)
    : result(setResult)
    , moves(setMoves.size())
    , mctsValues(setMctsValues)
{
    assert(setMoves.size() == setChildVisits.size());

    for (int i = 0; i < setMoves.size(); i++)
    {
        moves[i] = static_cast<uint16_t>(setMoves[i]);
    }

    // No point shrinking keys from 32 to 16 bits because they alternate with floats. Don't bother zipping/unzipping.
    childVisits = setChildVisits;

    moveCount = static_cast<int>(moves.size());
}

SavedGame::SavedGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<float>&& setMctsValues, std::vector<std::map<Move, float>>&& setChildVisits)
    : result(setResult)
    , moves(std::move(setMoves))
    , mctsValues(std::move(setMctsValues))
    , childVisits(std::move(setChildVisits))
{
    moveCount = static_cast<int>(moves.size());
}