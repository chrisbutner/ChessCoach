#include "Storage.h"

//#include <cassert>
//#include <cstdlib>
//#include <cstdio>
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
    , moveCount(static_cast<int>(setMoves.size()))
    , moves(setMoves.size())
{
    assert(setMoves.size() == setChildVisits.size());

    for (int i = 0; i < setMoves.size(); i++)
    {
        moves[i] = setMoves[i];
    }

    // No point shrinking keys from 32 to 16 bits because they alternate with floats. Don't bother zipping/unzipping.
    childVisits = setChildVisits;
}

Storage::Storage()
{
    char* rootEnvPath;
    errno_t err = _dupenv_s(&rootEnvPath, nullptr, RootEnvPath);
    assert(!err && rootEnvPath);

    _gamesPath = std::filesystem::path(rootEnvPath) / GamesPart;

    std::error_code error;
    std::filesystem::create_directories(_gamesPath, error);
    assert(!error);
}

void Storage::AddGame(const StoredGame&& game)
{
    int gameNumber;

    {
        std::lock_guard lock(_mutex);

        _games.emplace_back(game);
        gameNumber = static_cast<int>(_games.size());

        while (_games.size() > Config::WindowSize)
        {
            _games.pop_front();
        }
    }

    SaveToDisk(game, gameNumber);
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
    file << version << moveCount << game.result;

    file.write(reinterpret_cast<const char*>(game.moves.data()), sizeof(uint16_t) * moveCount);

    for (int i = 0; i < moveCount; i++)
    {
        const int mapSize = static_cast<int>(game.childVisits[i].size());
        file << mapSize;

        for (auto pair : game.childVisits[i])
        {
            file << pair.first << pair.second;
        }
    }
}

