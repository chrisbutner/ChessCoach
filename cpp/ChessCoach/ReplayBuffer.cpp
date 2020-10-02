#include "ReplayBuffer.h"

#include <cmath>

#include "Game.h"

Window ReplayBuffer::GetWindow() const
{
    return _window;
}

void ReplayBuffer::SetWindow(const Window& window)
{
    _window = window;
}

SavedGame& ReplayBuffer::AddGame(SavedGame&& game)
{
    for (auto map : game.childVisits)
    {
        for (auto [move, prior] : map)
        {
            assert(!std::isnan(prior));
        }
    }

    _gameMoveCounts.push_back(game.moveCount);
    _games.emplace_back(std::move(game));

    return _games.back();
}

int ReplayBuffer::GameCount() const
{
    return static_cast<int>(_games.size());
}

bool ReplayBuffer::SampleBatch(TrainingBatch& batch) const
{
    // Make sure that there are enough games ready to sample from.
    const int gamesInWindow = (std::min(_window.TrainingGameMax, static_cast<int>(_games.size())) - _window.TrainingGameMin);
    if (gamesInWindow < _window.MinimumSamplableGames)
    {
        return false;
    }

    std::discrete_distribution gameDistribution = CalculateGameDistribution();

    for (int i = 0; i < batch.images.size(); i++)
    {
        const int gameIndex = 
#if SAMPLE_BATCH_FIXED
            i;
#else
            (gameDistribution(Random::Engine) + _window.TrainingGameMin);
#endif
        const SavedGame& game = _games[gameIndex];

        const int positionIndex =
#if SAMPLE_BATCH_FIXED
            i % game.moveCount;
#else
            std::uniform_int_distribution<int>(0, game.moveCount - 1)(Random::Engine);
#endif

        // Populate the image, value and policy for the chosen position.
        Game scratchGame = _startingPosition;
        for (int m = 0; m < positionIndex; m++)
        {
            scratchGame.ApplyMove(Move(game.moves[m]));
        }

        batch.images[i] = scratchGame.GenerateImage();
        batch.values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
        batch.mctsValues[i] = game.mctsValues[positionIndex];
        batch.policies[i] = scratchGame.GeneratePolicy(game.childVisits[positionIndex]);

        // If there's a follow-up position then populate the reply policy. Otherwise, zero it.
        const int replyPositionIndex = (positionIndex + 1);
        if (replyPositionIndex < game.moveCount)
        {
            scratchGame.ApplyMove(Move(game.moves[replyPositionIndex - 1]));
            batch.replyPolicies[i] = scratchGame.GeneratePolicy(game.childVisits[replyPositionIndex]);
        }
        else
        {
            float* data = reinterpret_cast<float*>(batch.replyPolicies[i].data());
            std::fill(data, data + INetwork::OutputPlanesFloatCount, 0.f);
        }
    }

    return true;
}

std::discrete_distribution<int> ReplayBuffer::CalculateGameDistribution() const
{
    const int gameCount = static_cast<int>(_games.size());
    assert(_window.TrainingGameMin >= 0);
    assert(_window.TrainingGameMin < gameCount);
    assert(_window.TrainingGameMax >= 0);
    assert(_window.TrainingGameMin < _window.TrainingGameMax);
    const int trainingGameMax = std::min(gameCount, _window.TrainingGameMax);
    return std::discrete_distribution<int>(
        _gameMoveCounts.begin() + _window.TrainingGameMin,
        _gameMoveCounts.begin() + trainingGameMax);
}