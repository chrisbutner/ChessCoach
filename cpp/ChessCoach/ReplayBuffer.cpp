#include "ReplayBuffer.h"

#include <cmath>

#include "Game.h"

void ReplayBuffer::SetWindow(const Window& window)
{
    _window = window;

    // Recompute curriculum-window-skewed move counts for each game
    // to use as a sampling distribution.
    for (int i = 0; i < _games.size(); i++)
    {
        _gameMoveCounts[i] = CalculateMoveCount(_games[i]);
    }
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

    _gameMoveCounts.push_back(CalculateMoveCount(game));
    _games.emplace_back(std::move(game));

    return _games.back();
}

int ReplayBuffer::GameCount() const
{
    return static_cast<int>(_games.size());
}

bool ReplayBuffer::SampleBatch(TrainingBatch& batch) const
{
    // Use the training batch size as a rough minimum game count to sample from.
    const int gamesInWindow = (std::min(_window.TrainingGameMax, static_cast<int>(_games.size())) - _window.TrainingGameMin);
    if (gamesInWindow < static_cast<int>(batch.images.size()))
    {
        return false;
    }

    const std::discrete_distribution gameDistribution = CalculateGameDistribution();

    for (int i = 0; i < batch.images.size(); i++)
    {
        const int gameIndex = 
#if SAMPLE_BATCH_FIXED
            i;
#else
            (gameDistribution(Random::Engine) + _window.TrainingGameMin);
#endif
        const SavedGame& game = _games[gameIndex];

        const std::discrete_distribution positionDistribution = CalculatePositionDistribution(_games[gameIndex]);

        const int positionIndex =
#if SAMPLE_BATCH_FIXED
            i % game.moveCount;
#else
            positionDistribution(Random::Engine);
#endif

        // Populate the image, value and policy for the chosen position.
        Game scratchGame = _startingPosition;
        for (int m = 0; m < positionIndex; m++)
        {
            scratchGame.ApplyMove(Move(game.moves[m]));
        }

        batch.images[i] = scratchGame.GenerateImage();
        batch.values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
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

std::vector<Move> ReplayBuffer::SamplePartialGame(int minMovesBeforeEnd, int maxMovesBeforeEnd)
{
    while (true)
    {
        const int gameCount = static_cast<int>(_games.size());
        const int gameIndex = std::uniform_int_distribution<int>(0, gameCount - 1)(Random::Engine);
        const SavedGame& game = _games[gameIndex];
        if (game.moveCount > 512) // TODO: Access config
        {
            continue;
        }

        // There are N moves and (N+1) positions. Move M leads from position M to position (M+1).
        // The final position isn't useful because it's terminal, so (N-1) useful moves/positions
        const int positionIndex = game.moveCount - 1 -
            std::uniform_int_distribution<int>(minMovesBeforeEnd, maxMovesBeforeEnd)(Random::Engine);
        if (positionIndex < 0)
        {
            continue;
        }

        // For position 0 apply moves {}; for position 1 apply moves {0}; etc.
        std::vector<Move> moves(positionIndex);
        for (int m = 0; m < positionIndex; m++)
        {
            moves[m] = Move(game.moves[m]);
        }
        return moves;
    }
}

float ReplayBuffer::CalculateMoveCount(const SavedGame& game)
{
    const int positionCount = static_cast<int>(game.moveCount);
    const int endingPositionCount = std::min(positionCount, _window.CurriculumEndingPositions);
    return (_window.CurriculumEndingProbability * endingPositionCount) +
        ((1.f - _window.CurriculumEndingProbability) * (positionCount - endingPositionCount));
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

std::discrete_distribution<int> ReplayBuffer::CalculatePositionDistribution(const SavedGame& game) const
{
    const int positionCount = static_cast<int>(game.moveCount);
    std::vector<float> positionProbabilities(positionCount);
    assert(_window.CurriculumEndingPositions >= 0);
    assert(_window.CurriculumEndingProbability >= 0.f);
    assert(_window.CurriculumEndingProbability <= 1.f);

    // Distribute ending positions.
    const int endingPositionCount = std::min(positionCount, _window.CurriculumEndingPositions);
    const float endingPositionProbability = (_window.CurriculumEndingProbability / endingPositionCount);
    for (int j = positionCount - endingPositionCount; j < positionCount; j++)
    {
        positionProbabilities[j] = endingPositionProbability;
    }

    // Distribute early positions.
    const int earlyPositionCount = (positionCount - endingPositionCount);
    const float earlyPositionProbability = ((1.f - _window.CurriculumEndingProbability) / earlyPositionCount);
    for (int j = 0; j < earlyPositionCount; j++)
    {
        positionProbabilities[j] = earlyPositionProbability;
    }

    return std::discrete_distribution(positionProbabilities.begin(), positionProbabilities.end());
}