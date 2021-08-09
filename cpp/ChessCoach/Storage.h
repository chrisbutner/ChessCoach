// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#ifndef _STORAGE_H_
#define _STORAGE_H_

#include <filesystem>
#include <vector>
#include <atomic>
#include <mutex>

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

struct CommentaryRecordType
{
    std::filesystem::path directory;
    std::unique_ptr<message::Example> record;
    int latestRecordNumber = 0;
};

struct CommentarySaveContext
{
public:

    constexpr static const int PositionsPerRecord = 100000;

public:

    CommentaryRecordType& ChooseRecordType();

public:

    std::mutex mutex;
    std::vector<CommentaryRecordType> recordTypes;
    std::vector<float> recordWeights;
};

class Storage
{
public:

    static std::filesystem::path MakeLocalPath(const std::filesystem::path& path);

public:

    void FixChunk(INetwork* network, const std::string& filename);

    Storage();
    void InitializeLocalGamesChunks(INetwork* network);
    int AddTrainingGame(INetwork* network, SavedGame&& game);
    int TrainingGamesToPlay(int trainingChunkCount, int targetGameCount, bool ignoreLocalGames) const;

    void SaveChunk(const std::filesystem::path& path, const std::vector<SavedGame>& games) const;
    void SaveCommentary(CommentarySaveContext& saveContext, const std::vector<SavedGame>& games,
        std::vector<SavedCommentary>& gameCommentary, Vocabulary& vocabulary) const;
    void WriteRemainingCommentary(CommentarySaveContext& saveContext) const;
    std::string GenerateSimpleChunkFilename(int chunkNumber) const;

    void LoadGameFromChunk(const std::string& chunkContents, int gameIndex, SavedGame* gameOut);

    message::Example DebugPopulateGame(const SavedGame& game) const;
        
private:

    std::string GenerateFilename(int number);
    void TryChunkMultiple(INetwork* network);
    void ChunkGames(INetwork* network, std::vector<std::filesystem::path>& gamePaths);
    void PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut) const;
    void WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message) const;
    uint32_t MaskCrc32cForTfRecord(uint32_t crc32c) const;
    bool SkipTfRecord(google::protobuf::io::ZeroCopyInputStream& stream) const;
    void WriteCommentary(CommentaryRecordType& recordType) const;

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
    std::filesystem::path _relativePgnsPath;
};

#endif // _STORAGE_H_