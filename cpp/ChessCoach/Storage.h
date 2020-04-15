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
#include "SavedGame.h"

enum GameType
{
    GameType_Train,
    GameType_Test,
    GameType_Supervised,

    GameType_Count,
};

constexpr const char* GameTypeNames[GameType_Count] = { "Train", "Test", "Supervised" };

struct TrainingBatch
{
    std::array<INetwork::InputPlanes, Config::BatchSize> images;
    std::array<float, Config::BatchSize> values;
    std::array<INetwork::OutputPlanes, Config::BatchSize> policies;
};

class Storage
{
private:

    static constexpr const char* const RootEnvPath = "localappdata";
    static constexpr const char* const GamesPart = "ChessCoach/Training/Games";
    static constexpr const char* const TrainPart = "Train";
    static constexpr const char* const TestPart = "Test";
    static constexpr const char* const SupervisedPart = "Supervised";
    static constexpr const char* const NetworksPart = "ChessCoach/Training/Networks";
    static constexpr const char* const LogsPart = "ChessCoach/Training/Logs";

public:

    static SavedGame LoadFromDisk(const std::filesystem::path& path);
    static void SaveToDisk(const std::filesystem::path& path, const SavedGame& game);
    static std::string GenerateGameName(int gameNumber);

public:

    Storage();
    Storage(const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
        const std::filesystem::path& supervisedTestPath, const std::filesystem::path& networksPath);

    void LoadExistingGames(GameType gameType, int maxLoadCount);
    int AddGame(GameType gameType, SavedGame&& game);
    TrainingBatch* SampleBatch(GameType gameType);
    int GamesPlayed(GameType gameType) const;
    int CountNetworks() const;
    std::filesystem::path LogPath() const;
        
private:

    std::pair<const SavedGame*, int> AddGameWithoutSaving(GameType gameType, SavedGame&& game);
    void SaveToDisk(GameType gameType, const SavedGame& game, int gameNumber) const;

private:

    mutable std::mutex _mutex;
    std::deque<SavedGame> _games;
    std::array<int, GameType_Count> _latestGameNumbers;

    std::unique_ptr<TrainingBatch> _trainingBatch;
    Game _startingPosition;

    mutable std::default_random_engine _random;

    std::array<std::filesystem::path, GameType_Count> _gamesPaths;
    std::filesystem::path _networksPath;
    std::filesystem::path _logsPath;
};

#endif // _STORAGE_H_