#include "Storage.h"

#include <string>
#include <sstream>
#include <iostream>
#include <chrono>

#include "Config.h"
#include "Pgn.h"
#include "Platform.h"
#include "Random.h"

using namespace std::chrono_literals;

Storage::Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _pipelines{ { { _games[GameType_Training], _gameMoveCounts[GameType_Training], networkConfig.Training.BatchSize },
        { _games[GameType_Validation], _gameMoveCounts[GameType_Validation], networkConfig.Training.BatchSize } } }
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _trainingWindowSize(networkConfig.Training.WindowSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
{
    _trainingBatchSize = networkConfig.Training.BatchSize;
    _pgnInterval = networkConfig.Training.PgnInterval;

    const std::filesystem::path rootPath = Platform::UserDataPath();

    static_assert(GameType_Count == 2);
    _gamesPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.GamesPathTraining);
    _gamesPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.GamesPathValidation);
    _pgnsPath = MakePath(rootPath, miscConfig.Paths_Pgns);
    _networksPath = MakePath(rootPath, miscConfig.Paths_Networks);
    _logsPath = MakePath(rootPath, miscConfig.Paths_Logs);

    for (std::filesystem::path gamesPath : _gamesPaths)
    {
        std::filesystem::create_directories(gamesPath);
    }
    std::filesystem::create_directories(_pgnsPath);
    std::filesystem::create_directories(_networksPath);
    std::filesystem::create_directories(_logsPath);

    InitializePipelines();
}

// Used for testing
Storage::Storage(const NetworkConfig& networkConfig,
    const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
    const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _pipelines{ { { _games[GameType_Training], _gameMoveCounts[GameType_Training], networkConfig.Training.BatchSize },
        { _games[GameType_Validation], _gameMoveCounts[GameType_Validation], networkConfig.Training.BatchSize } } }
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _trainingWindowSize(networkConfig.Training.WindowSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
    , _gamesPaths{ gamesTrainPath, gamesTestPath }
    , _pgnsPath(pgnsPath)
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 2);

    InitializePipelines();
}

void Storage::InitializePipelines()
{
    static_assert(GameType_Count == 2);
    _pipelines[GameType_Training].StartWorkers(2);
    _pipelines[GameType_Validation].StartWorkers(1);
}

void Storage::LoadExistingGames(GameType gameType, int maxLoadCount)
{
    // Should be no contention at initialization time. Just bulk lock around
    // (a) AddGameWithoutSaving, and (b) setting _gameFileCount/_loadedGameCount.
    std::lock_guard lock(_mutex);

    const int gameFileCount = static_cast<int>(std::distance(std::filesystem::directory_iterator(_gamesPaths[gameType]), std::filesystem::directory_iterator()));

    int lastPrintedCount = 0;
    int loadedCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
    {
        const int maxAdditionalCount = (maxLoadCount - loadedCount);
        const int justLoadedCount = LoadFromDiskInternal(entry.path(),
            std::bind(&Storage::AddGameWithoutSaving, this, gameType, std::placeholders::_1),
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
    
    AddGameWithoutSaving(gameType, std::move(game));

    const std::deque<SavedGame>& games = _games[gameType];
    const SavedGame& emplaced = games.back();

    SaveToDisk(gameType, emplaced);

    const int gameNumber = _loadedGameCount[gameType];
    
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

// Requires caller to lock.
void Storage::AddGameWithoutSaving(GameType gameType, SavedGame&& game)
{
    std::deque<SavedGame>& games = _games[gameType];
    std::deque<int>& gameMoveCounts = _gameMoveCounts[gameType];

    for (auto map : game.childVisits)
    {
        for (auto [move, prior] : map)
        {
            assert(!std::isnan(prior));
        }
    }

    gameMoveCounts.push_back(game.moveCount);
    games.emplace_back(std::move(game));
    
    ++_loadedGameCount[gameType];

    // TODO: Popping for window size removed until pipeline threading accounts for it.
}

// No locking: assumes single-threaded sampling periods without games being played.
TrainingBatch* Storage::SampleBatch(GameType gameType)
{
    return _pipelines[gameType].SampleBatch();
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

SavedGame Storage::LoadSingleGameFromDisk(const std::filesystem::path& path)
{
    SavedGame game;
    LoadFromDiskInternal(path, [&](SavedGame&& loaded) { game = std::move(loaded); }, 1);
    return game;
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

void Storage::SaveSingleGameToDisk(const std::filesystem::path& path, const SavedGame& game)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const Version version = Version1;
    const GameCount gameCount = 1;

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&gameCount), sizeof(gameCount));

    SaveToDiskInternal(file, game);
}

void Storage::SaveMultipleGamesToDisk(const std::filesystem::path& path, const std::vector<SavedGame>& games)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const Version version = 1;
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
    const MoveCount moveCount = static_cast<int>(game.moves.size());

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
    // Root any relative paths at ChessCoach's appdata directory.
    if (path.is_absolute())
    {
        return path;
    }
    return (root / path);
}

Pipeline::Pipeline(const std::deque<SavedGame>& games, const std::deque<int>& gameMoveCounts, int trainingBatchSize)
    : _games(&games)
    , _gameMoveCounts(&gameMoveCounts)
    , _trainingBatchSize(trainingBatchSize)
{
}

void Pipeline::StartWorkers(int workerCount)
{
    for (int i = 0; i < workerCount; i++)
    {
        // Just leak them until clean ending/joining for the whole training process is implemented.
        new std::thread(&Pipeline::GenerateBatches, this);
    }
}

TrainingBatch* Pipeline::SampleBatch()
{
    std::unique_lock lock(_mutex);

    while (_count <= 0)
    {
        //std::cout << "SampleBatch pipeline starved" << std::endl;
        _batchExists.wait(lock);
    }

    const int index = ((_oldest + BufferCount - _count) % BufferCount);
    if (--_count == (MaxFill - 1))
    {
        _roomExists.notify_one();
    }
    return &_batches[index];
}

void Pipeline::AddBatch(TrainingBatch&& batch)
{
    std::unique_lock lock(_mutex);

    while (_count >= MaxFill)
    {
        _roomExists.wait(lock);
    }

    _batches[_oldest] = std::move(batch);

    _oldest = ((_oldest + 1) % BufferCount);
    if (++_count == 1)
    {
        _batchExists.notify_one();
    }
}

void Pipeline::GenerateBatches()
{
    const std::deque<SavedGame>& games = *_games;
    const std::deque<int>& gameMoveCounts = *_gameMoveCounts;
    Game startingPosition;
    TrainingBatch workingBatch;
    workingBatch.images.resize(_trainingBatchSize);
    workingBatch.values.resize(_trainingBatchSize);
    workingBatch.policies.resize(_trainingBatchSize);

    // Use _trainingBatchSize as a rough minimum game count to sample from.
    while (games.size() < _trainingBatchSize)
    {
        std::this_thread::sleep_for(1s);
    }

    while (true)
    {
        const int gameCount = static_cast<int>(games.size());

        std::discrete_distribution distribution(gameMoveCounts.begin(), gameMoveCounts.begin() + gameCount);

        for (int i = 0; i < _trainingBatchSize; i++)
        {
            const SavedGame& game =
#if SAMPLE_BATCH_FIXED
                games[i];
#else
                games[distribution(Random::Engine)];
#endif

            int positionIndex =
#if SAMPLE_BATCH_FIXED
                i % game.moveCount;
#else
                std::uniform_int_distribution<int>(0, game.moveCount - 1)(Random::Engine);
#endif

            Game scratchGame = startingPosition;
            for (int m = 0; m < positionIndex; m++)
            {
                scratchGame.ApplyMove(Move(game.moves[m]));
            }

            workingBatch.images[i] = scratchGame.GenerateImage();
            workingBatch.values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
            workingBatch.policies[i] = scratchGame.GeneratePolicy(game.childVisits[positionIndex]);
        }

        AddBatch(std::move(workingBatch));
        workingBatch.images.resize(_trainingBatchSize);
        workingBatch.values.resize(_trainingBatchSize);
        workingBatch.policies.resize(_trainingBatchSize);
    }
}