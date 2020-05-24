#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <mutex>
#include <deque>
#include <map>
#include <vector>
#include <functional>
#include <fstream>

#include <Stockfish/position.h>

#include "Network.h"
#include "Game.h"
#include "SavedGame.h"

enum GameType
{
    GameType_Training,
    GameType_Validation,

    GameType_Count,
};

constexpr const char* GameTypeNames[GameType_Count] = { "Training", "Validation" };

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
};

class Pipeline
{
public:

    // Don't completely fill the buffer, to avoid clobbbering the in-use sampled batch.
    static const int BufferCount = 16;
    static const int MaxFill = (BufferCount - 1);

public:

    Pipeline(const std::deque<SavedGame>& games, const std::deque<int>& gameMoveCounts, int trainingBatchSize);
    void StartWorkers(int workerCount);
    TrainingBatch* SampleBatch();

private:

    void GenerateBatches();
    void AddBatch(TrainingBatch&& batch);

private:

    const std::deque<SavedGame>* _games;
    const std::deque<int>* _gameMoveCounts;
    int _trainingBatchSize;
    std::array<TrainingBatch, BufferCount> _batches;
    std::mutex _mutex;
    std::condition_variable _batchExists;
    std::condition_variable _roomExists;

    int _oldest = 0;
    int _count = 0;
};

class Storage
{
private:


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

    Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig);
    Storage(const NetworkConfig& networkConfig,
        const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
        const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath);

    void LoadExistingGames(GameType gameType, int maxLoadCount);
    int AddGame(GameType gameType, SavedGame&& game);
    TrainingBatch* SampleBatch(GameType gameType);
    int GamesPlayed(GameType gameType) const;
    int NetworkStepCount(const std::string& networkName) const;
    std::filesystem::path LogPath() const;
        
private:

    void InitializePipelines();
    void AddGameWithoutSaving(GameType gameType, SavedGame&& game);
    void SaveToDisk(GameType gameType, const SavedGame& game);
    std::filesystem::path MakePath(const std::filesystem::path& root, const std::filesystem::path& path);

private:

    mutable std::mutex _mutex;
    std::array<std::deque<SavedGame>, GameType_Count> _games;
    std::array<std::deque<int>, GameType_Count> _gameMoveCounts;
    std::array<int, GameType_Count> _gameFileCount;
    std::array<int, GameType_Count> _loadedGameCount;

    std::array<std::ofstream, GameType_Count> _currentSaveFile;
    std::array<int, GameType_Count> _currentSaveGameCount;

    std::array<Pipeline, GameType_Count> _pipelines;

    int _trainingBatchSize;
    int _trainingWindowSize;
    int _pgnInterval;

    std::array<std::filesystem::path, GameType_Count> _gamesPaths;
    std::filesystem::path _pgnsPath;
    std::filesystem::path _networksPath;
    std::filesystem::path _logsPath;
};

#endif // _STORAGE_H_