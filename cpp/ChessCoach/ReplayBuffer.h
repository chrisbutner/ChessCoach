#ifndef _REPLAYBUFFER_H_
#define _REPLAYBUFFER_H_

#include <deque>

#include "SavedGame.h"
#include "Game.h"
#include "Random.h"

class ReplayBuffer
{
public:

    Window GetWindow() const;
    void SetWindow(const Window& window);
    SavedGame& AddGame(SavedGame&& game);
    int GameCount() const;
    bool SampleBatch(TrainingBatch& batch) const;
    std::vector<Move> SamplePartialGame(int maxMoves, int minMovesBeforeEnd, int maxMovesBeforeEnd);

private:

    float CalculateMoveCount(const SavedGame& game);
    std::discrete_distribution<int> CalculateGameDistribution() const;
    std::discrete_distribution<int> CalculatePositionDistribution(const SavedGame& game) const;

private:

    Window _window = {};
    Game _startingPosition;
    std::deque<SavedGame> _games;
    std::deque<float> _gameMoveCounts;
};

#endif // _REPLAYBUFFER_H_