#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <mutex>
#include <deque>
#include <random>
#include <map>
#include <vector>

#include <Stockfish/Position.h>

#include "Network.h"
#include "Game.h"

struct TrainingBatch
{
    std::array<INetwork::InputPlanes, Config::BatchSize> images;
    std::array<float, Config::BatchSize> values;
    std::array<INetwork::OutputPlanes, Config::BatchSize> policies;
};

struct StoredGame
{
public:

    StoredGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::map<Move, float>>& setChildVisits);
    StoredGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<std::map<Move, float>>&& setChildVisits);

    float result;
    int moveCount;
    std::vector<uint16_t> moves;
    std::vector<std::map<Move, float>> childVisits;
};

class Storage
{
private:

    static constexpr const char* const RootEnvPath = "localappdata";
    static constexpr const char* const GamesPart = "ChessCoach/Training/Games";
    static constexpr const char* const NetworksPart = "ChessCoach/Training/Networks";

public:

    Storage();
    Storage(const std::filesystem::path& gamesPath, const std::filesystem::path& networksPath);

    void LoadExistingGames();
    int AddGame(StoredGame&& game);
    TrainingBatch* SampleBatch();
    int GamesPlayed() const;
    int CountNetworks() const;
    StoredGame LoadFromDisk(const std::string& path) const;
        
private:

    StoredGame& AddGameWithoutSaving(StoredGame&& game);
    void SaveToDisk(const StoredGame& game, int gameNumber) const;

private:

    mutable std::mutex _mutex;
    std::deque<StoredGame> _games;
    int _latestGameNumber;

    std::unique_ptr<TrainingBatch> _trainingBatch;
    Game _startingPosition;

    mutable std::default_random_engine _random;

    std::filesystem::path _gamesPath;
    std::filesystem::path _networksPath;
};

#endif // _STORAGE_H_