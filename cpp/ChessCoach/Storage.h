#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <mutex>
#include <deque>
#include <map>
#include <vector>
#include <functional>
#include <fstream>
#include <atomic>

#include <Stockfish/position.h>

#include "Network.h"
#include "Game.h"
#include "SavedGame.h"

namespace google {
    namespace protobuf {
        class Message;
        namespace io {
            class ZeroCopyOutputStream;
        }
    }
}

namespace message {
    class Example;
}

class Storage
{
public:

    static void SaveChunk(const Game& startingPosition, const std::filesystem::path& path, const std::vector<SavedGame>& games);
    static std::string GenerateChunkFilename(int chunkNumber);

public:

    Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig);

    int AddGame(GameType gameType, SavedGame&& game);
    int GamesPlayed(GameType gameType) const;
    std::filesystem::path LogPath() const;
        
private:

    void SaveGame(GameType gameType, const SavedGame& game, int gameNumber);
    static void PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut);
    static void WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message);
    static uint32_t MaskCrc32cForTfRecord(uint32_t crc32c);
    std::filesystem::path MakePath(const std::filesystem::path& root, const std::filesystem::path& path);
    void LoadCommentary();

private:

    mutable std::mutex _mutex;

    SavedCommentary _commentary;
    const Game _startingPosition;

    int _pgnInterval;

    std::string _vocabularyFilename;
    std::string _sessionPrefix;
    std::atomic_int _sessionGameCount;
    std::array<std::filesystem::path, GameType_Count> _gamesPaths;
    std::array<std::filesystem::path, GameType_Count> _commentaryPaths;
    std::filesystem::path _pgnsPath;
    std::filesystem::path _logsPath;
};

#endif // _STORAGE_H_