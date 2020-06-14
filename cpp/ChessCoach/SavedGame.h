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
    SavedGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::map<Move, float>>& setChildVisits);
    SavedGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<std::map<Move, float>>&& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<std::map<Move, float>> childVisits;
};

struct TrainingBatch
{
    TrainingBatch() = default;
    ~TrainingBatch() = default;
    TrainingBatch(const TrainingBatch& other) = delete;
    TrainingBatch& operator=(const TrainingBatch& other) = delete;
    TrainingBatch(TrainingBatch&& other) noexcept = default;
    TrainingBatch& operator=(TrainingBatch&& other) noexcept = default;

    std::vector<INetwork::InputPlanes> images;
    std::vector<float> values;
    std::vector<INetwork::OutputPlanes> policies;
    std::vector<INetwork::OutputPlanes> replyPolicies;
};

struct Window
{
    // E.g. for 5000 games per network, with current window size of 10000, on network #4,
    // set TrainingGameMin=10000, TrainingGameMax=20000.
    int TrainingGameMin; // Inclusive, 0-based
    int TrainingGameMax; // Exclusive, 0-based

    // E.g. for 60% probability of sampling one of the final 5 positions, or 40% in the earlier positions,
    // set CurriculumEndingPositions=5, CurriculumEndingProbability=0.6f. For an even distribution, set 0/0.f.
    int CurriculumEndingPositions;
    float CurriculumEndingProbability;
};

#endif // _SAVEDGAME_H_