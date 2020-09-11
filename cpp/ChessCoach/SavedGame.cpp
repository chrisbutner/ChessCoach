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
        moves[i] = setMoves[i];
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

SavedComment::SavedComment()
    : gameIndex(-1)
    , moveIndex(-1)
{
}

SavedComment::SavedComment(int setGameIndex, int setMoveIndex, std::vector<uint16_t>&& setVariationMoves, std::string&& setComment)
    : gameIndex(setGameIndex)
    , moveIndex(setMoveIndex)
    , variationMoves(std::move(setVariationMoves))
    , comment(std::move(setComment))
{
}