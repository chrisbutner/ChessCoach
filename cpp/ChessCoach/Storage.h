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
#include "ReplayBuffer.h"
#include "Pipeline.h"

enum GameType
{
    GameType_Training,
    GameType_Validation,

    GameType_Count,
};

constexpr const char* GameTypeNames[GameType_Count] = { "Training", "Validation" };
static_assert(GameType_Count == 2);

class Storage
{
private:


    using Version = uint16_t;
    using GameCount = uint16_t;
    using MoveCount = uint16_t;

    static const Version Version1 = 1;

public:

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
        const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesValidationPath,
        const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath);

    void LoadExistingGames(GameType gameType, int maxLoadCount);
    int AddGame(GameType gameType, SavedGame&& game);
    TrainingBatch* SampleBatch(GameType gameType);
    int GamesPlayed(GameType gameType) const;
    int NetworkStepCount(const std::string& networkName) const;
    std::filesystem::path LogPath() const;
    Window GetWindow(GameType gameType) const;
    void SetWindow(GameType gameType, const Window& window);
    CommentaryTrainingBatch* SampleCommentaryBatch();
        
private:

    void InitializePipelines(const NetworkConfig& networkConfig);
    void SaveToDisk(GameType gameType, const SavedGame& game);
    std::filesystem::path MakePath(const std::filesystem::path& root, const std::filesystem::path& path);
    void LoadCommentary();

private:

    mutable std::mutex _mutex;
    std::array<ReplayBuffer, GameType_Count> _games;
    std::array<int, GameType_Count> _gameFileCount;
    std::array<int, GameType_Count> _loadedGameCount;

    std::array<std::ofstream, GameType_Count> _currentSaveFile;
    std::array<int, GameType_Count> _currentSaveGameCount;

    std::array<Pipeline, GameType_Count> _pipelines;

    SavedCommentary _commentary;
    CommentaryTrainingBatch _commentaryBatch;
    Game _startingPosition;

    int _trainingBatchSize;
    int _trainingCommentaryBatchSize;
    int _pgnInterval;

    std::string _vocabularyFilename;
    std::array<std::filesystem::path, GameType_Count> _gamesPaths;
    std::array<std::filesystem::path, GameType_Count> _commentaryPaths;
    std::filesystem::path _pgnsPath;
    std::filesystem::path _networksPath;
    std::filesystem::path _logsPath;
};

#endif // _STORAGE_H_