#include "Storage.h"

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#include "Config.h"

StoredGame::StoredGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::map<Move, float>>& setChildVisits)
    : result(setResult)
    , moves(setMoves.size())
{
    assert(setMoves.size() == setChildVisits.size());

    for (int i = 0; i < setMoves.size(); i++)
    {
        moves[i] = setMoves[i];
    }

    // No point shrinking keys from 32 to 16 bits because they alternate with floats. Don't bother zipping/unzipping.
    childVisits = setChildVisits;

    moveCount = static_cast<int>(moves.size());
}

StoredGame::StoredGame(float setResult, std::vector<uint16_t>&& setMoves, std::vector<std::map<Move, float>>&& setChildVisits)
    : result(setResult)
    , moves(std::move(setMoves))
    , childVisits(std::move(setChildVisits))
{
    moveCount = static_cast<int>(moves.size());
}

Storage::Storage()
    : _latestGameNumber(0)
    , _random(std::random_device()() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
    char* rootEnvPath;
    errno_t err = _dupenv_s(&rootEnvPath, nullptr, RootEnvPath);
    assert(!err && rootEnvPath);

    _gamesPath = std::filesystem::path(rootEnvPath) / GamesPart;
    _networksPath = std::filesystem::path(rootEnvPath) / NetworksPart;

    std::error_code error;
    std::filesystem::create_directories(_gamesPath, error);
    assert(!error);
}

void Storage::LoadExistingGames()
{
#ifdef _DEBUG
    std::cout << "Skipping game loading in debug" << std::endl;
#else
    for (const auto& directory : std::filesystem::directory_iterator(_gamesPath))
    {
        std::cout << "Loading game: " << directory.path().filename() << std::endl;
        AddGameWithoutSaving(LoadFromDisk(directory.path().string()));
    }
#endif
}

int Storage::AddGame(StoredGame&& game)
{
    const StoredGame& emplaced = AddGameWithoutSaving(std::move(game));
    
    const int gameNumber = _latestGameNumber;
    SaveToDisk(emplaced, gameNumber);

    return gameNumber;
}

StoredGame& Storage::AddGameWithoutSaving(StoredGame&& game)
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

        StoredGame& emplaced = _games.emplace_back(std::move(game));

        // It would be nice to use _games.size() but we pop_front() beyond the window size.
        _latestGameNumber++;

        while (_games.size() > Config::WindowSize)
        {
            _games.pop_front();
        }

        return emplaced;
    }
}

TrainingBatch* Storage::SampleBatch()
{
    int positionSum = 0;

    if (!_trainingBatch)
    {
        _trainingBatch.reset(new TrainingBatch());
    }

    for (const StoredGame& game : _games)
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
        const StoredGame& game = _games[distribution(_random)];

        int positionIndex = std::uniform_int_distribution<int>(0, game.moveCount - 1)(_random);

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

int Storage::GamesPlayed() const
{
    std::lock_guard lock(_mutex);

    // Starts at 0, incremented with each game added.
    return _latestGameNumber;
}

int Storage::CountNetworks() const
{
    return static_cast<int>(std::distance(std::filesystem::directory_iterator(_networksPath.string()), std::filesystem::directory_iterator()));
}

void Storage::SaveToDisk(const StoredGame& game, int gameNumber) const
{
    std::filesystem::path gamePath = _gamesPath;

    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << gameNumber;

    gamePath /= "game_";
    gamePath += suffix.str();

    std::ofstream file = std::ofstream(gamePath, std::ios::out | std::ios::binary);

    const uint16_t version = 1;
    const uint16_t moveCount = static_cast<int>(game.moves.size());

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
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

StoredGame Storage::LoadFromDisk(const std::string& path) const
{
    std::ifstream file = std::ifstream(path, std::ios::in | std::ios::binary);

    uint16_t version;
    uint16_t moveCount;
    float result;

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&moveCount), sizeof(moveCount));
    file.read(reinterpret_cast<char*>(&result), sizeof(result));
    assert(version == 1);
    assert(moveCount >= 1);
    assert((result >= 0.f) && (result <= 1.f));

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

    return StoredGame(result, std::move(moves), std::move(childVisits));
}