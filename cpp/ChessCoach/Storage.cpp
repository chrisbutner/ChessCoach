#include "Storage.h"

#include <fstream>
#include <string>
#include <sstream>

#include "Config.h"

TrainingBatch::TrainingBatch(InputPlanes* setImages, float* setValues, OutputPlanes* setPolicies)
    : images(setImages)
    , values(setValues)
    , policies(setPolicies)
{
}

TrainingBatch::~TrainingBatch()
{
    delete[] images;
    delete[] values;
    delete[] policies;
}

StoredGame::StoredGame(float setResult, const std::vector<Move>& setMoves, const std::vector<std::unordered_map<Move, float>>& setChildVisits)
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

StoredGame::StoredGame(float setResult, const std::vector<uint16_t>&& setMoves, const std::vector<std::unordered_map<Move, float>>&& setChildVisits)
    : result(setResult)
    , moves(setMoves)
    , childVisits(setChildVisits)
{
    moveCount = static_cast<int>(moves.size());
}

Storage::Storage()
    : _nextGameNumber(1)
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

#include <iostream>
void Storage::LoadExistingGames()
{
    for (const auto& directory : std::filesystem::directory_iterator(_gamesPath))
    {
        std::cout << "Loading game: " << directory.path().filename() << std::endl;
        AddGameWithoutSaving(LoadFromDisk(directory.path().string()));
    }
}

int Storage::AddGame(StoredGame&& game)
{
    int gameNumber = AddGameWithoutSaving(std::move(game));

    SaveToDisk(game, gameNumber);

    return gameNumber;
}

int Storage::AddGameWithoutSaving(StoredGame&& game)
{
    int gameNumber;

    for (auto map : game.childVisits)
    {
        for (auto pair : map)
        {
            assert(!std::isnan(pair.second));
        }
    }

    {
        std::lock_guard lock(_mutex);

        _games.emplace_back(game);

        // It would be nice to use _games.size() but we pop_front() beyond the window size.
        gameNumber = _nextGameNumber++;

        while (_games.size() > Config::WindowSize)
        {
            _games.pop_front();
        }
    }

    return gameNumber;
}

TrainingBatch Storage::SampleBatch() const
{
    int positionSum = 0;

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

    InputPlanes* images = new InputPlanes[Config::BatchSize];
    float* values = new float[Config::BatchSize];
    OutputPlanes* policies = new OutputPlanes[Config::BatchSize];

    for (int i = 0; i < Config::BatchSize; i++)
    {
        const StoredGame& game = _games[distribution(_random)];

        int positionIndex = std::uniform_int_distribution<int>(0, game.moveCount - 1)(_random);

        Game scratchGame = _startingPosition;
        for (int m = 0; m < positionIndex; m++)
        {
            scratchGame.ApplyMove(Move(game.moves[m]));
        }

        images[i] = scratchGame.GenerateImage();
        values[i] = Game::FlipValue(scratchGame.ToPlay(), game.result);
        policies[i] = scratchGame.GeneratePolicy(game.childVisits[positionIndex]);
    }

    return TrainingBatch(images, values, policies);
}

int Storage::GamesPlayed() const
{
    std::lock_guard lock(_mutex);

    // Starts at 1
    return (_nextGameNumber - 1);
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

StoredGame Storage::LoadFromDisk(std::string path) const
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

    std::vector<std::unordered_map<Move, float>> childVisits(moveCount);
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