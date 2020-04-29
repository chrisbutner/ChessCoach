#include "Storage.h"

#include <string>
#include <sstream>
#include <iostream>

#include "Config.h"
#include "Pgn.h"

Storage::Storage()
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _random(std::random_device()() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
    char* rootEnvPath;
    errno_t err = _dupenv_s(&rootEnvPath, nullptr, RootEnvPath);
    assert(!err && rootEnvPath);

    static_assert(GameType_Count == 3);
    _gamesPaths[GameType_Train] = std::filesystem::path(rootEnvPath) / GamesPart / TrainPart;
    _gamesPaths[GameType_Test] = std::filesystem::path(rootEnvPath) / GamesPart / TestPart;
    _gamesPaths[GameType_Supervised] = std::filesystem::path(rootEnvPath) / GamesPart / SupervisedPart;
    _pgnsPath = std::filesystem::path(rootEnvPath) / PgnsPart;
    _networksPath = std::filesystem::path(rootEnvPath) / NetworksPart;
    _logsPath = std::filesystem::path(rootEnvPath) / LogsPart;

    for (std::filesystem::path gamesPath : _gamesPaths)
    {
        std::filesystem::create_directories(gamesPath);
    }
    std::filesystem::create_directories(_pgnsPath);
    std::filesystem::create_directories(_networksPath);
    std::filesystem::create_directories(_logsPath);
}

// Used for testing
Storage::Storage(const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
    const std::filesystem::path& supervisedTestPath, const std::filesystem::path& pgnsPath,
    const std::filesystem::path& networksPath)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _random(std::random_device()() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()))
    , _gamesPaths{ gamesTrainPath, gamesTestPath, supervisedTestPath }
    , _pgnsPath(pgnsPath)
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 3);
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

int Storage::AddGame(GameType gameType, SavedGame&& game, const NetworkConfig& networkConfig)
{
    std::lock_guard lock(_mutex);
    
    AddGameWithoutSaving(gameType, std::move(game));

    const std::deque<SavedGame>& games = _games[gameType];
    const SavedGame& emplaced = games.back();
    SaveToDisk(gameType, emplaced);

    const int gameNumber = _loadedGameCount[gameType];
    
    if ((gameNumber % networkConfig.Training.PgnInterval) == 0)
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

    for (auto map : game.childVisits)
    {
        for (auto [move, prior] : map)
        {
            assert(!std::isnan(prior));
        }
    }

    games.emplace_back(std::move(game));
    ++_loadedGameCount[gameType];
}

// No locking: assumes single-threaded sampling periods without games being played.
TrainingBatch* Storage::SampleBatch(GameType gameType, const NetworkConfig& networkConfig)
{
    int positionSum = 0;
    std::deque<SavedGame>& games = _games[gameType];

    if (_trainingBatch.images.size() != networkConfig.Training.BatchSize)
    {
        _trainingBatch.images.resize(networkConfig.Training.BatchSize);
        _trainingBatch.values.resize(networkConfig.Training.BatchSize);
        _trainingBatch.policies.resize(networkConfig.Training.BatchSize);
    }

    while (games.size() > networkConfig.Training.WindowSize)
    {
        games.pop_front();
    }

    for (const SavedGame& game : games)
    {
        positionSum += game.moveCount;
    }

    std::vector<float> probabilities(games.size());
    for (int i = 0; i < games.size(); i++)
    {
        probabilities[i] = static_cast<float>(games[i].moveCount) / positionSum;
    }

    std::discrete_distribution distribution(probabilities.begin(), probabilities.end());

    for (int i = 0; i < networkConfig.Training.BatchSize; i++)
    {
        const SavedGame& game =
#if SAMPLE_BATCH_FIXED
            games[i];
#else
            games[distribution(_random)];
#endif

        int positionIndex =
#if SAMPLE_BATCH_FIXED
            i % game.moveCount;
#else
            std::uniform_int_distribution<int>(0, game.moveCount - 1)(_random);
#endif

        Game scratchGame = _startingPosition;
        for (int m = 0; m < positionIndex; m++)
        {
            scratchGame.ApplyMove(Move(game.moves[m]));
        }

        _trainingBatch.images[i] = scratchGame.GenerateImage();
        _trainingBatch.values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
        _trainingBatch.policies[i] = scratchGame.GeneratePolicy(game.childVisits[positionIndex]);
    }

    return &_trainingBatch;
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
    bool startNewFile = false;

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