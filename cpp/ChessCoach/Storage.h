#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <mutex>
#include <deque>
#include <random>
#include <map>
#include <vector>
#include <functional>
#include <fstream>

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
    std::vector<INetwork::InputPlanes> images;
    std::vector<float> values;
    std::vector<INetwork::OutputPlanes> policies;
};

class Storage
{
private:

    static constexpr const char* const RootEnvPath = "localappdata";
    static constexpr const char* const GamesPart = "ChessCoach/Training/Games";
    static constexpr const char* const TrainPart = "Train";
    static constexpr const char* const TestPart = "Test";
    static constexpr const char* const SupervisedPart = "Supervised";
    static constexpr const char* const PgnsPart = "ChessCoach/Training/Pgns";
    static constexpr const char* const NetworksPart = "ChessCoach/Training/Networks";
    static constexpr const char* const LogsPart = "ChessCoach/Training/Logs";

    using Version = uint16_t;
    using GameCount = uint16_t;
    using MoveCount = uint16_t;

    static const Version Version1 = 1;

public:

    static SavedGame LoadSingleGameFromDisk(const std::filesystem::path& path);
    static void SaveSingleGameToDisk(const std::filesystem::path& path, const SavedGame& game);
    static void SaveMultipleGamesToDisk(const std::filesystem::path& path, const std::vector<SavedGame>& games);
    static std::string GenerateGamesFilename(int gamesNumber);

private:

    static int LoadFromDiskInternal(const std::filesystem::path& path,
        std::function<void(SavedGame&&)> gameHandler, int maxLoadCount);
    static void SaveToDiskInternal(std::ofstream& file, const SavedGame& game, GameCount newGameCountInFile);
    static void SaveToDiskInternal(std::ofstream& file, const SavedGame& game);

public:

    Storage();
    Storage(const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
        const std::filesystem::path& supervisedTestPath, const std::filesystem::path& pgnsPath,
        const std::filesystem::path& networksPath);

    void LoadExistingGames(GameType gameType, int maxLoadCount);
    int AddGame(GameType gameType, SavedGame&& game, const NetworkConfig& networkConfig);
    TrainingBatch* SampleBatch(GameType gameType, const NetworkConfig& networkConfig);
    int GamesPlayed(GameType gameType) const;
    int NetworkStepCount(const std::string& networkName) const;
    std::filesystem::path LogPath() const;
        
private:

    void AddGameWithoutSaving(GameType gameType, SavedGame&& game);
    void SaveToDisk(GameType gameType, const SavedGame& game);

private:

    mutable std::mutex _mutex;
    std::array<std::deque<SavedGame>, GameType_Count> _games;
    std::array<int, GameType_Count> _gameFileCount;
    std::array<int, GameType_Count> _loadedGameCount;

    std::array<std::ofstream, GameType_Count> _currentSaveFile;
    std::array<int, GameType_Count> _currentSaveGameCount;

    TrainingBatch _trainingBatch;
    Game _startingPosition;

    mutable std::default_random_engine _random;

    std::array<std::filesystem::path, GameType_Count> _gamesPaths;
    std::filesystem::path _pgnsPath;
    std::filesystem::path _networksPath;
    std::filesystem::path _logsPath;
};

#endif // _STORAGE_H_