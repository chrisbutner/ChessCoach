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

private:

    std::discrete_distribution<int> CalculateGameDistribution() const;

private:

    Window _window = { 0, 0, std::numeric_limits<int>::max() };
    Game _startingPosition;
    std::deque<SavedGame> _games;
    std::deque<int> _gameMoveCounts;
};

#endif // _REPLAYBUFFER_H_