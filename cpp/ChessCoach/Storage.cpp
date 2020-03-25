#include "Storage.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>

Storage::Storage()
    : _gameNumber(1)
{
    char* rootEnvPath;
    errno_t err = _dupenv_s(&rootEnvPath, nullptr, RootEnvPath);
    assert(!err && rootEnvPath);

    _gamesPath = std::filesystem::path(rootEnvPath) / GamesPart;

    std::error_code error;
    std::filesystem::create_directories(_gamesPath, error);
    assert(!error);
}

void Storage::SaveToDisk(const StoredGame& game) const
{
    errno_t openError;
    FILE* file;

    do
    {
        std::filesystem::path gamePath = _gamesPath;
        
        std::stringstream suffix;
        suffix << std::setfill('0') << std::setw(9) << _gameNumber;

        gamePath /= "game_";
        gamePath += suffix.str();

        openError = fopen_s(&file, gamePath.string().c_str(), "wxb");
        _gameNumber++;
    } while (openError);

    const int version = 1;
    fwrite(reinterpret_cast<const void*>(&version), sizeof(version), 1, file);

    fwrite(reinterpret_cast<const void*>(&game.terminalValue), sizeof(game.terminalValue), 1, file);

    const int moveCount = static_cast<int>(game.moves.size());
    fwrite(reinterpret_cast<const void*>(&moveCount), sizeof(moveCount), 1, file);

    fwrite(reinterpret_cast<const void*>(game.moves.data()), sizeof(int), moveCount, file);
    fwrite(reinterpret_cast<const void*>(game.images.data()), sizeof(float), moveCount * 12 * 8 * 8, file);
    fwrite(reinterpret_cast<const void*>(game.policies.data()), sizeof(int), moveCount * 73 * 8 * 8, file);

    int closeError = fclose(file);
    assert(!closeError);
}

