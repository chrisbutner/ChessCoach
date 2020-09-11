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
    std::vector<float> mctsValues;
    std::vector<INetwork::OutputPlanes> policies;
    std::vector<INetwork::OutputPlanes> replyPolicies;
};

struct CommentaryTrainingBatch
{
    std::vector<INetwork::InputPlanes> images;
    std::vector<std::string> comments;
};

struct Window
{
    // E.g. for 5000 games per network, with current window size of 10000, on network #4,
    // set TrainingGameMin=10000, TrainingGameMax=20000.
    int TrainingGameMin; // Inclusive, 0-based
    int TrainingGameMax; // Exclusive, 0-based
    int MinimumSamplableGames; // Usually roughly training batch size
};

#endif // _SAVEDGAME_H_