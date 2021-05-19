#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <vector>
#include <atomic>

#include "Network.h"
#include "Game.h"
#include "SavedGame.h"

namespace google {
    namespace protobuf {
        class Message;
        namespace io {
            class ZeroCopyInputStream;
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

    Storage();
    void InitializeLocalGamesChunks(INetwork* network);
    int AddTrainingGame(INetwork* network, SavedGame&& game);
    int TrainingGamesToPlay(int trainingChunkCount, int targetGameCount, bool ignoreLocalGames) const;
    std::filesystem::path LocalLogPath() const;

    void SaveChunk(const std::filesystem::path& path, const std::vector<SavedGame>& games) const;
    void SaveCommentary(const std::filesystem::path& path, const std::vector<SavedGame>& games,
        std::vector<SavedCommentary>& gameCommentary, Vocabulary& vocabulary) const;
    std::string GenerateSimpleChunkFilename(int chunkNumber) const;

    void LoadGameFromChunk(const std::string& chunkContents, int gameIndex, SavedGame* gameOut);

    message::Example DebugPopulateGame(const SavedGame& game) const;
        
private:

    std::string GenerateFilename(int number);
    void TryChunkMultiple(INetwork* network);
    void ChunkGames(INetwork* network, std::vector<std::filesystem::path>& gamePaths);
    void PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut) const;
    void WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message) const;
    static uint32_t MaskCrc32cForTfRecord(uint32_t crc32c);
    bool SkipTfRecord(google::protobuf::io::ZeroCopyInputStream& stream) const;
    std::filesystem::path MakeLocalPath(const std::filesystem::path& root, const std::filesystem::path& path);

    template <typename T>
    bool Read(google::protobuf::io::ZeroCopyInputStream& stream, T& value) const;

private:

    const Game _startingPosition;

    std::atomic_int _trainingGameCount;
    int _gamesPerChunk;

    int _pgnInterval;

    std::string _sessionNonce;
    std::atomic_int _sessionGameCount;
    std::atomic_int _sessionChunkCount;
    std::filesystem::path _relativeTrainingGamePath;
    std::filesystem::path _localTrainingGamePath;
    std::filesystem::path _localLogsPath;
    std::filesystem::path _relativePgnsPath;
};

#endif // _STORAGE_H_