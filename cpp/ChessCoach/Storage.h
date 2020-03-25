#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>

#include "Network.h"

struct StoredGame
{
public:

    StoredGame(float terminalValue, size_t moveCount);

    float terminalValue;
    std::vector<int> moves;
    std::vector<InputPlanes> images;
    std::vector<OutputPlanes> policies;
};

class Storage
{
private:

    static constexpr const char* const RootEnvPath = "localappdata";
    static constexpr const char* const GamesPart = "ChessCoach/Training/Games";

public:

    Storage();

    void SaveToDisk(const StoredGame& game) const;
        
private:

    std::filesystem::path _gamesPath;
    mutable int _gameNumber;
};

#endif // _STORAGE_H_