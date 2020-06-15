#include "Storage.h"

#include <string>
#include <sstream>
#include <iostream>
#include <chrono>

#include "Config.h"
#include "Pgn.h"
#include "Platform.h"

Storage::Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
    , _maxMoves(networkConfig.SelfPlay.MaxMoves)
{
    _trainingBatchSize = networkConfig.Training.BatchSize;
    _pgnInterval = networkConfig.Training.PgnInterval;

    const std::filesystem::path rootPath = Platform::UserDataPath();

    static_assert(GameType_Count == 3);
    _gamesPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.GamesPathTraining);
    _gamesPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.GamesPathValidation);
    _gamesPaths[GameType_Curriculum] = MakePath(rootPath, networkConfig.Training.GamesPathCurriculum);
    _pgnsPath = MakePath(rootPath, miscConfig.Paths_Pgns);
    _networksPath = MakePath(rootPath, miscConfig.Paths_Networks);
    _logsPath = MakePath(rootPath, miscConfig.Paths_Logs);

    InitializePipelines(networkConfig);
}

// Used for testing
Storage::Storage(const NetworkConfig& networkConfig,
    const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath, const std::filesystem::path& gamesCurriculumPath,
    const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
    , _maxMoves(networkConfig.SelfPlay.MaxMoves)
    , _gamesPaths{ gamesTrainPath, gamesTestPath, gamesCurriculumPath }
    , _pgnsPath(pgnsPath)
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 3);

    InitializePipelines(networkConfig);
}

void Storage::InitializePipelines(const NetworkConfig& networkConfig)
{
    static_assert(GameType_Count == 3);
    _pipelines[GameType_Training].Initialize(_games[GameType_Training], networkConfig.Training.BatchSize, 2 /* workerCount */);
    _pipelines[GameType_Validation].Initialize(_games[GameType_Validation], networkConfig.Training.BatchSize, 1 /* workerCount */);
    // No curriculum pipeline
}

void Storage::LoadExistingGames(GameType gameType, int maxLoadCount)
{
    // Should be no contention at initialization time. Just bulk lock around
    // (a) AddGameWithoutSaving, and (b) setting _gameFileCount/_loadedGameCount.
    std::lock_guard lock(_mutex);

    if (_gamesPaths[gameType].empty())
    {
        return;
    }

    ReplayBuffer& games = _games[gameType];
    const int gameFileCount = static_cast<int>(std::distance(std::filesystem::directory_iterator(_gamesPaths[gameType]), std::filesystem::directory_iterator()));

    int lastPrintedCount = 0;
    int loadedCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
    {
        const int maxAdditionalCount = (maxLoadCount - loadedCount);
        const int justLoadedCount = LoadFromDiskInternal(entry.path(),
            std::bind(&ReplayBuffer::AddGame, &games, std::placeholders::_1),
            maxAdditionalCount);
        loadedCount += justLoadedCount;

        const int printCount = (loadedCount - (loadedCount % 1000));
        if (printCount > lastPrintedCount)
        {
            std::cout << printCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
            lastPrintedCount = printCount;
        }
        if (loadedCount >= maxLoadCount)
        {
            break;
        }
    }

    if (loadedCount > lastPrintedCount)
    {
        std::cout << loadedCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
        lastPrintedCount = loadedCount;
    }

    _gameFileCount[gameType] = gameFileCount;
    _loadedGameCount[gameType] = loadedCount;
}

int Storage::AddGame(GameType gameType, SavedGame&& game)
{
    std::lock_guard lock(_mutex);
    
    const SavedGame& emplaced = _games[gameType].AddGame(std::move(game));

    SaveToDisk(gameType, emplaced);

    const int gameNumber = ++_loadedGameCount[gameType];
    
    if ((gameNumber % _pgnInterval) == 0)
    {
        std::stringstream suffix;
        suffix << std::setfill('0') << std::setw(9) << gameNumber;
        const std::string& filename = ("game_" + suffix.str() + ".pgn");
        const std::filesystem::path pgnPath = _pgnsPath / filename;

        std::ofstream pgnFile = std::ofstream(pgnPath, std::ios::out);
        Pgn::GeneratePgn(pgnFile, emplaced);
    }

    return gameNumber;
}

// No locking: assumes single-threaded sampling periods without games being played.
TrainingBatch* Storage::SampleBatch(GameType gameType)
{
    return _pipelines[gameType].SampleBatch();
}

std::vector<Move> Storage::SamplePartialGame(int minPlyBeforeEnd, int maxPlyBeforeEnd)
{
    return _games[GameType_Curriculum].SamplePartialGame(_maxMoves, minPlyBeforeEnd, maxPlyBeforeEnd);
}

int Storage::GamesPlayed(GameType gameType) const
{
    std::lock_guard lock(_mutex);

    return _loadedGameCount[gameType];
}

int Storage::NetworkStepCount(const std::string& networkName) const
{
    const std::string prefix = networkName + "_";
    std::filesystem::directory_entry lastEntry;
    for (const auto& entry : std::filesystem::directory_iterator(_networksPath))
    {
        if (entry.path().filename().string().compare(0, prefix.size(), prefix) == 0)
        {
            lastEntry = entry;
        }
    }

    if (lastEntry.path().empty())
    {
        return 0;
    }

    std::stringstream tokenizer(lastEntry.path().filename().string());
    std::string ignore;
    int networkStepCount;
    std::getline(tokenizer, ignore, '_');
    tokenizer >> networkStepCount;

    return networkStepCount;
}

// Requires caller to lock.
void Storage::SaveToDisk(GameType gameType, const SavedGame& game)
{
    // Check whether the current save file has run out of room.
    if (_currentSaveFile[gameType].is_open())
    {
        if (_currentSaveGameCount[gameType] >= Config::Misc.Storage_MaxGamesPerFile)
        {
            _currentSaveFile[gameType] = {};
            _currentSaveGameCount[gameType] = 0;
        }
    }
    // Try to continue from a previous run, but only add to the final file, not any "inside" gaps.
    // This should only run once; future SaveToDisk calls should always see an open _currentSaveFile[gameType].
    else
    {
        std::filesystem::directory_entry lastEntry;
        for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
        {
            lastEntry = entry;
        }

        std::ifstream lastFile;
        if (!lastEntry.path().empty() && (lastFile = std::ifstream(lastEntry.path(), std::ios::in | std::ios::binary)))
        {
            Version version;
            GameCount gameCount;

            lastFile.read(reinterpret_cast<char*>(&version), sizeof(version));
            lastFile.read(reinterpret_cast<char*>(&gameCount), sizeof(gameCount));
            assert(version == Version1);

            if (gameCount < Config::Misc.Storage_MaxGamesPerFile)
            {
                lastFile.close();
                _currentSaveFile[gameType] = std::ofstream(lastEntry.path(), std::ios::in | std::ios::out | std::ios::binary);
                _currentSaveGameCount[gameType] = gameCount;
            }
        }
    }

    // We may need to start a new file.
    if (!_currentSaveFile[gameType].is_open())
    {
        const int newGameFileNumber = ++_gameFileCount[gameType];
        std::filesystem::path gamesPath = _gamesPaths[gameType] / GenerateGamesFilename(newGameFileNumber);
        _currentSaveFile[gameType] = std::ofstream(gamesPath, std::ios::out | std::ios::binary);
        _currentSaveGameCount[gameType] = 0;

        const Version version = Version1;
        const GameCount initialGameCount = 0;
        _currentSaveFile[gameType].write(reinterpret_cast<const char*>(&version), sizeof(version));
        _currentSaveFile[gameType].write(reinterpret_cast<const char*>(&initialGameCount), sizeof(initialGameCount));
    }

    // Save the game and flush to disk.
    const GameCount newGameCountInFile = ++_currentSaveGameCount[gameType];
    SaveToDiskInternal(_currentSaveFile[gameType], game, newGameCountInFile);
    _currentSaveFile[gameType].flush();
}

int Storage::LoadFromDiskInternal(const std::filesystem::path& path, std::function<void(SavedGame&&)> gameHandler, int maxLoadCount)
{
    std::ifstream file = std::ifstream(path, std::ios::in | std::ios::binary);

    Version version;
    GameCount gameCount;

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&gameCount), sizeof(gameCount));
    assert(version == Version1);

    const int loadCount = std::min(maxLoadCount, static_cast<int>(gameCount));
    for (int i = 0; i < loadCount; i++)
    {
        MoveCount moveCount;
        float result;

        file.read(reinterpret_cast<char*>(&moveCount), sizeof(moveCount));
        file.read(reinterpret_cast<char*>(&result), sizeof(result));
        assert(moveCount >= 1);
        assert((result == CHESSCOACH_VALUE_WIN) ||
            (result == CHESSCOACH_VALUE_DRAW) ||
            (result == CHESSCOACH_VALUE_LOSS));

        std::vector<uint16_t> moves(moveCount);
        file.read(reinterpret_cast<char*>(moves.data()), sizeof(uint16_t) * moveCount);

        std::vector<std::map<Move, float>> childVisits(moveCount);
        for (int i = 0; i < moveCount; i++)
        {
            int mapSize;
            file.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));
            for (int j = 0; j < mapSize; j++)
            {
                Move key;
                float value;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                childVisits[i][key] = value;
            }
        }

        gameHandler(SavedGame(result, std::move(moves), std::move(childVisits)));
    }

    return loadCount;
}

void Storage::SaveMultipleGamesToDisk(const std::filesystem::path& path, const std::vector<SavedGame>& games)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const Version version = Version1;
    const GameCount gameCount = static_cast<GameCount>(games.size());
    assert(games.size() <= std::numeric_limits<GameCount>::max());

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&gameCount), sizeof(gameCount));

    for (const SavedGame& game : games)
    {
        SaveToDiskInternal(file, game);
    }
}

void Storage::SaveToDiskInternal(std::ofstream& file, const SavedGame& game, GameCount newGameCountInFile)
{
    assert(newGameCountInFile <= std::numeric_limits<GameCount>::max());

    // Update the game count.
    file.seekp(sizeof(Version), file.beg);
    file.write(reinterpret_cast<const char*>(&newGameCountInFile), sizeof(newGameCountInFile));
    file.seekp(0, file.end);

    // Write the game.
    SaveToDiskInternal(file, game);
}


void Storage::SaveToDiskInternal(std::ofstream& file, const SavedGame& game)
{
    // Write the game.
    const MoveCount moveCount = static_cast<MoveCount>(game.moves.size());

    file.write(reinterpret_cast<const char*>(&moveCount), sizeof(moveCount));
    file.write(reinterpret_cast<const char*>(&game.result), sizeof(game.result));

    file.write(reinterpret_cast<const char*>(game.moves.data()), sizeof(uint16_t) * moveCount);

    for (int i = 0; i < moveCount; i++)
    {
        const int mapSize = static_cast<int>(game.childVisits[i].size());
        file.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

        for (auto pair : game.childVisits[i])
        {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        }
    }
}

std::string Storage::GenerateGamesFilename(int gamesNumber)
{
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << gamesNumber;

    return "games_" + suffix.str();
}

std::filesystem::path Storage::LogPath() const
{
    return _logsPath;
}

std::filesystem::path Storage::MakePath(const std::filesystem::path& root, const std::filesystem::path& path)
{
    // Empty paths have special meaning as N/A.
    if (path.empty())
    {
        return path;
    }

    // Root any relative paths at ChessCoach's appdata directory.
    if (path.is_absolute())
    {
        std::filesystem::create_directories(path);
        return path;
    }

    const std::filesystem::path rooted = (root / path);
    std::filesystem::create_directories(rooted);
    return rooted;
}

Window Storage::GetWindow(GameType gameType) const
{
    return _games[gameType].GetWindow();
}

void Storage::SetWindow(GameType gameType, const Window& window)
{
    std::cout << "Setting window: " << window.TrainingGameMin << " -> " << window.TrainingGameMax
        << ", " << window.CurriculumEndingProbability << " @ last " << window.CurriculumEndingPositions << " positions ("
        << GameTypeNames[gameType] << ")" << std::endl;
    _games[gameType].SetWindow(window);
}