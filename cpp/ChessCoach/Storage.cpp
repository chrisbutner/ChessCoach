#include "Storage.h"

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#include "Config.h"

Storage::Storage()
    : _latestGameNumbers{}
    , _random(std::random_device()() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
    char* rootEnvPath;
    errno_t err = _dupenv_s(&rootEnvPath, nullptr, RootEnvPath);
    assert(!err && rootEnvPath);

    static_assert(GameType_Count == 3);
    _gamesPaths[GameType_Train] = std::filesystem::path(rootEnvPath) / GamesPart / TrainPart;
    _gamesPaths[GameType_Test] = std::filesystem::path(rootEnvPath) / GamesPart / TestPart;
    _gamesPaths[GameType_Supervised] = std::filesystem::path(rootEnvPath) / GamesPart / SupervisedPart;
    _networksPath = std::filesystem::path(rootEnvPath) / NetworksPart;
    _logsPath = std::filesystem::path(rootEnvPath) / LogsPart;

    for (std::filesystem::path gamesPath : _gamesPaths)
    {
        std::filesystem::create_directories(gamesPath);
    }
    std::filesystem::create_directories(_networksPath);
    std::filesystem::create_directories(_logsPath);
}

Storage::Storage(const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesTestPath,
    const std::filesystem::path& supervisedTestPath, const std::filesystem::path& networksPath)
    : _latestGameNumbers{}
    , _random(std::random_device()() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()))
    , _gamesPaths{ gamesTrainPath, gamesTestPath, supervisedTestPath }
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 3);
}

void Storage::LoadExistingGames(GameType gameType, int maxLoadCount)
{
    // Accesses _latestGameNumbers without locking (avoids having to re-enter)
    // but only uses it to print the number of games loaded.
    const int foundCount = static_cast<int>(std::distance(std::filesystem::directory_iterator(_gamesPaths[gameType]), std::filesystem::directory_iterator()));
    int loadedCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
    {
        const int maxAdditionalCount = (maxLoadCount - loadedCount);
        const int justLoadedCount = LoadFromDiskInternal(entry.path(),
            std::bind(&Storage::AddGameWithoutSaving, this,gameType, std::placeholders::_1),
            maxAdditionalCount);
        loadedCount += justLoadedCount;
        if (((loadedCount % 1000) == 0) || (justLoadedCount > 1))
        {
            std::cout << loadedCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
        }
        if (loadedCount >= maxLoadCount)
        {
            std::cout << loadedCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
            std::cout << (foundCount - loadedCount) << " games skipped (" << GameTypeNames[gameType] << ")" << std::endl;

            {
                std::lock_guard lock(_mutex);

                _latestGameNumbers[gameType] = foundCount;
            }

            return;
        }
    }
    std::cout << loadedCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
}

int Storage::AddGame(GameType gameType, SavedGame&& game)
{
    auto [emplaced, gameNumber] = AddGameWithoutSaving(gameType, std::move(game));

    SaveToDisk(gameType, *emplaced, gameNumber);

    return gameNumber;
}

std::pair<const SavedGame*, int> Storage::AddGameWithoutSaving(GameType gameType, SavedGame&& game)
{
    for (auto map : game.childVisits)
    {
        for (auto [move, prior] : map)
        {
            assert(!std::isnan(prior));
        }
    }

    {
        std::lock_guard lock(_mutex);

        SavedGame* emplaced = &_games.emplace_back(std::move(game));

        // It would be nice to use _games.size() but we pop_front() beyond the window size.
        const int gameNumber = ++_latestGameNumbers[gameType];

        while (_games.size() > Config::WindowSize)
        {
            _games.pop_front();
        }

        return std::pair(emplaced, gameNumber);
    }
}

TrainingBatch* Storage::SampleBatch(GameType gameType)
{
    int positionSum = 0;

    if (!_trainingBatch)
    {
        _trainingBatch.reset(new TrainingBatch());
    }

    for (const SavedGame& game : _games)
    {
        positionSum += game.moveCount;
    }

    std::vector<float> probabilities(_games.size());
    for (int i = 0; i < _games.size(); i++)
    {
        probabilities[i] = static_cast<float>(_games[i].moveCount) / positionSum;
    }

    std::discrete_distribution distribution(probabilities.begin(), probabilities.end());

    for (int i = 0; i < Config::BatchSize; i++)
    {
        const SavedGame& game =
#if SAMPLE_BATCH_FIXED
            _games[i];
#else
            _games[distribution(_random)];
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

        _trainingBatch->images[i] = scratchGame.GenerateImage();
        _trainingBatch->values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
        _trainingBatch->policies[i] = scratchGame.GeneratePolicy(game.childVisits[positionIndex]);
    }

    return _trainingBatch.get();
}

int Storage::GamesPlayed(GameType gameType) const
{
    std::lock_guard lock(_mutex);

    // Starts at 0, incremented with each game added.
    return _latestGameNumbers[gameType];
}

int Storage::CountNetworks() const
{
    return static_cast<int>(std::distance(std::filesystem::directory_iterator(_networksPath.string()), std::filesystem::directory_iterator()));
}

void Storage::SaveToDisk(GameType gameType, const SavedGame& game, int gameNumber) const
{
    std::filesystem::path gamePath = _gamesPaths[gameType];
    gamePath /= GenerateGamesFilename(gameNumber);

    SaveToDisk(gamePath, game);
}

SavedGame Storage::LoadFromDisk(const std::filesystem::path& path)
{
    SavedGame game;
    LoadFromDiskInternal(path, [&](SavedGame&& loaded) { game = std::move(loaded); }, 1);
    return game;
}

int Storage::LoadFromDiskInternal(const std::filesystem::path& path, std::function<void(SavedGame&&)> gameHandler, int maxLoadCount)
{
    std::ifstream file = std::ifstream(path, std::ios::in | std::ios::binary);

    uint16_t version;
    uint16_t gameCount;

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&gameCount), sizeof(gameCount));
    assert(version == 1);
    assert(gameCount >= 1);

    const int loadCount = std::min(maxLoadCount, static_cast<int>(gameCount));
    for (int i = 0; i < loadCount; i++)
    {
        uint16_t moveCount;
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

void Storage::SaveToDisk(const std::filesystem::path& path, const SavedGame& game)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const uint16_t version = 1;
    const uint16_t gameCount = 1;

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&gameCount), sizeof(gameCount));

    SaveToDiskInternal(file, game);
}

void Storage::SaveToDisk(const std::filesystem::path& path, const std::vector<SavedGame>& games)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const uint16_t version = 1;
    const uint16_t gameCount = static_cast<uint16_t>(games.size());
    assert(games.size() <= std::numeric_limits<uint16_t>::max());

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&gameCount), sizeof(gameCount));

    for (const SavedGame& game : games)
    {
        SaveToDiskInternal(file, game);
    }
}

void Storage::SaveToDiskInternal(std::ofstream& file, const SavedGame& game)
{
    const uint16_t moveCount = static_cast<int>(game.moves.size());

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