#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <mutex>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

#include <Stockfish/Position.h>

#include "Network.h"
#include "Game.h"

struct TrainingBatch
{
    TrainingBatch(InputPlanes* images, float* values, OutputPlanes* policies);
    ~TrainingBatch();

    InputPlanes* images;
    float* values;
    OutputPlanes* policies;
};

struct StoredGame
{
public:

    StoredGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::unordered_map<Move, float>>& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<std::unordered_map<Move, float>> childVisits;
};

class Storage
{
private:

    static constexpr const char* const RootEnvPath = "localappdata";
    static constexpr const char* const GamesPart = "ChessCoach/Training/Games";

public:

    Storage();

    void AddGame(const StoredGame&& game);
    TrainingBatch SampleBatch() const;
        
private:

    void SaveToDisk(const StoredGame& game, int gameNumber) const;

private:

    std::mutex _mutex;
    std::deque<StoredGame> _games;
    Game _startingPosition;

    mutable std::default_random_engine _random;

    std::filesystem::path _gamesPath;
};

#endif // _STORAGE_H_