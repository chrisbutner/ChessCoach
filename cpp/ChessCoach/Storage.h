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
    TrainingBatch(INetwork::InputPlanes* images, float* values, INetwork::OutputPlanes* policies);
    ~TrainingBatch();

    INetwork::InputPlanes* images;
    float* values;
    INetwork::OutputPlanes* policies;
};

struct StoredGame
{
public:

    StoredGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::unordered_map<Move, float>>& setChildVisits);
    StoredGame(float setResult, const std::vector<uint16_t>&& setMoves, const std::vector<std::unordered_map<Move, float>>&& setChildVisits);

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
    static constexpr const char* const NetworksPart = "ChessCoach/Training/Networks";

public:

    Storage();

    void LoadExistingGames();
    int AddGame(StoredGame&& game);
    TrainingBatch SampleBatch() const;
    int GamesPlayed() const;
    int CountNetworks() const;
    StoredGame LoadFromDisk(const std::string& path) const;
        
private:

    int AddGameWithoutSaving(StoredGame&& game);
    void SaveToDisk(const StoredGame& game, int gameNumber) const;

private:

    mutable std::mutex _mutex;
    std::deque<StoredGame> _games;
    int _nextGameNumber;

    Game _startingPosition;

    mutable std::default_random_engine _random;

    std::filesystem::path _gamesPath;
    std::filesystem::path _networksPath;
};

#endif // _STORAGE_H_