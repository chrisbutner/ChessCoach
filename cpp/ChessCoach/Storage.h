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
    static std::string GenerateSimpleChunkFilename(int chunkNumber);

public:

    Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig, int trainingChunkCount);
    void Housekeep(INetwork* network);
    int AddTrainingGame(INetwork* network, SavedGame&& game);
    int TrainingGamesToPlay(int targetCount) const;
    std::filesystem::path LocalLogPath() const;
        
private:

    std::string GenerateFilename(int number);
    void TryChunk(INetwork* network);
    void ChunkGames(INetwork* network, std::vector<std::filesystem::path>& gamePaths);
    static void PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut);
    static void WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message);
    static uint32_t MaskCrc32cForTfRecord(uint32_t crc32c);
    std::filesystem::path MakeLocalPath(const std::filesystem::path& root, const std::filesystem::path& path);

private:

    mutable std::mutex _mutex;

    SavedCommentary _commentary;
    const Game _startingPosition;

    std::atomic_int _trainingChunkCount;
    std::atomic_int _trainingGameCount;
    int _gamesPerChunk;

    int _pgnInterval;

    std::string _vocabularyFilename;
    std::string _sessionNonce;
    std::atomic_int _sessionGameCount;
    std::atomic_int _sessionChunkCount;
    std::filesystem::path _relativeTrainingGamePath;
    std::filesystem::path _localTrainingGamePath;
    //std::array<std::filesystem::path, GameType_Count> _commentaryPaths;
    std::filesystem::path _localLogsPath;
    std::filesystem::path _relativePgnsPath;
};

#endif // _STORAGE_H_