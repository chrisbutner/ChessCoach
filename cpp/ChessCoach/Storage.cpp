#include "Storage.h"

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>

#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <protobuf/ChessCoach.pb.h>

#include <crc32c/include/crc32c/crc32c.h>

#include "Config.h"
#include "Pgn.h"
#include "Platform.h"
#include "Preprocessing.h"
#include "Random.h"

Storage::Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig)
    : _pgnInterval(networkConfig.Training.PgnInterval)
    , _vocabularyFilename(networkConfig.Training.VocabularyFilename)
{
    const std::filesystem::path rootPath = Platform::UserDataPath();

    static_assert(GameType_Count == 3);
    _gamesPaths[GameType_Supervised] = MakePath(rootPath, networkConfig.Training.GamesPathSupervised);
    _gamesPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.GamesPathTraining);
    _gamesPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.GamesPathValidation);
    _commentaryPaths[GameType_Supervised] = MakePath(rootPath, networkConfig.Training.CommentaryPathSupervised);
    _commentaryPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.CommentaryPathTraining);
    _commentaryPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.CommentaryPathValidation);
    _pgnsPath = MakePath(rootPath, miscConfig.Paths_Pgns);
    _networksPath = MakePath(rootPath, miscConfig.Paths_Networks);
    _logsPath = MakePath(rootPath, miscConfig.Paths_Logs);

    _sessionPrefix = "SESSION_"; // TODO: hostname plus GUID or timestamp
}

// Used for testing
Storage::Storage(const NetworkConfig& networkConfig, const std::filesystem::path& gamesSupervisedPath,
    const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesValidationPath,
    const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath)
    : _pgnInterval(networkConfig.Training.PgnInterval)
    , _vocabularyFilename() // TODO: Update with commentary unit tests
    , _gamesPaths{ gamesSupervisedPath, gamesTrainPath, gamesValidationPath }
    , _commentaryPaths { "", "", "" } // TODO: Update with commentary unit tests
    , _pgnsPath(pgnsPath)
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 3);

    _sessionPrefix = "SESSION_"; // TODO: hostname plus GUID or timestamp
}

int Storage::AddGame(GameType gameType, SavedGame&& game)
{
    const int gameNumber = ++_sessionGameCount;
    SaveGame(gameType, game, gameNumber);

    if ((gameNumber % _pgnInterval) == 0)
    {
        std::stringstream suffix;
        suffix << std::setfill('0') << std::setw(9) << gameNumber;
        const std::string filename = _sessionPrefix + suffix.str() + ".pgn";
        const std::filesystem::path pgnPath = _pgnsPath / filename;

        std::ofstream pgnFile = std::ofstream(pgnPath, std::ios::out);
        Pgn::GeneratePgn(pgnFile, game);
    }

    return gameNumber;
}

int Storage::GamesPlayed(GameType gameType) const
{
    // TODO: until chunking. should use OS-specific call to avoid iteration unless it doesn't.
    return static_cast<int>(std::distance(std::filesystem::directory_iterator(_gamesPaths[gameType]), std::filesystem::directory_iterator()));
}

int Storage::NetworkStepCount(const std::string& networkName) const
{
    const std::string prefix = networkName + "_";
    std::filesystem::directory_entry lastEntry;
    for (const auto& entry : std::filesystem::directory_iterator(_networksPath))
    {
        if (entry.path().filename().string().compare(0, prefix.size(), prefix) == 0)
        {
            lastEntry = entry;
        }
    }

    if (lastEntry.path().empty())
    {
        return 0;
    }

    std::stringstream tokenizer(lastEntry.path().filename().string());
    std::string ignore;
    int networkStepCount;
    std::getline(tokenizer, ignore, '_');
    tokenizer >> networkStepCount;

    return networkStepCount;
}

std::string Storage::GenerateChunkFilename(int chunkNumber)
{
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << chunkNumber;

    return suffix.str() + ".chunk";
}
void Storage::SaveGame(GameType gameType, const SavedGame& game, int gameNumber)
{
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << gameNumber;
    const std::string filename = _sessionPrefix + suffix.str() + ".game";
    const std::filesystem::path gamePath = _gamesPaths[gameType] / filename;

    SaveChunk(_startingPosition, gamePath, { game });
}

void Storage::SaveChunk(const Game& startingPosition, const std::filesystem::path& path, const std::vector<SavedGame>& games)
{
    // Compress the TFRecord file using zlib.
    std::ofstream file(path, std::ios::out | std::ios::binary);
    google::protobuf::io::OstreamOutputStream wrapped(&file);
    google::protobuf::io::GzipOutputStream::Options zipOptions{};
    zipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
    google::protobuf::io::GzipOutputStream zip(&wrapped, zipOptions);

    // Write an "Example" protobuf for each game as a TFRecord.
    message::Example storeGame;
    std::string buffer;
    for (const SavedGame& game : games)
    {
        PopulateGame(startingPosition, game, storeGame);
        WriteTfRecord(zip, buffer, storeGame);
    }

    if (!file.good())
    {
        throw std::runtime_error("Failed to write to file: " + path.string());
    }
}

void Storage::PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut)
{
    auto& features = *gameOut.mutable_features()->mutable_feature();

    // Write result directly.
    auto& result = *features["result"].mutable_float_list()->mutable_value();
    result.Clear();
    result.Add(game.result);

    // Write MCTS values directly.
    auto& mctsValues = *features["mcts_values"].mutable_float_list()->mutable_value();
    mctsValues.Clear();
    mctsValues.Reserve(game.moveCount);
    mctsValues.AddNAlreadyReserved(game.moveCount);
    std::copy(game.mctsValues.begin(), game.mctsValues.end(), mctsValues.mutable_data());

    // Fix up result and MCTS value.
    // MCTS deals with probabilities in [0, 1]. Network deals with tanh outputs/targets in (-1, 1)/[-1, 1].
    INetwork::MapProbabilities01To11(result.size(), result.mutable_data());
    INetwork::MapProbabilities01To11(mctsValues.size(), mctsValues.mutable_data());

    // Image and policy require applying moves to a scratch game, so process a move-at-once.
    // Policy indices/values are ragged, so reserve for each move.
    auto& imagePiecesAuxiliary = *features["image_pieces_auxiliary"].mutable_int64_list()->mutable_value();
    imagePiecesAuxiliary.Clear();
    const int imagePiecesAuxiliaryStride = (INetwork::InputPiecePlanesPerPosition + INetwork::InputAuxiliaryPlaneCount);
    const int imagePiecesAuxiliaryTotalSize = (game.moveCount * imagePiecesAuxiliaryStride);
    imagePiecesAuxiliary.Reserve(imagePiecesAuxiliaryTotalSize);
    imagePiecesAuxiliary.AddNAlreadyReserved(imagePiecesAuxiliaryTotalSize);

    auto& policyRowLengths = *features["policy_row_lengths"].mutable_int64_list()->mutable_value();
    policyRowLengths.Clear();
    policyRowLengths.Reserve(game.moveCount);
    policyRowLengths.AddNAlreadyReserved(game.moveCount);

    auto& policyIndices = *features["policy_indices"].mutable_int64_list()->mutable_value();
    policyIndices.Clear();
    auto& policyValues = *features["policy_values"].mutable_float_list()->mutable_value();
    policyValues.Clear();

    for (int m = 0; m < game.moveCount; m++)
    {
        INetwork::PackedPlane* imagePiecesOut = reinterpret_cast<INetwork::PackedPlane*>(imagePiecesAuxiliary.mutable_data()) + (m * imagePiecesAuxiliaryStride);
        INetwork::PackedPlane* imageAuxiliaryOut = (imagePiecesOut + INetwork::InputPiecePlanesPerPosition);
        scratchGame.GenerateImageCompressed(imagePiecesOut, imageAuxiliaryOut);

        const int movePolicyIndexCount = static_cast<int>(game.childVisits[m].size());
        policyRowLengths[m] = movePolicyIndexCount;

        const int cumulativePolicyIndexCountOld = policyIndices.size();
        const int cumulativePolicyIndexCountNew = (cumulativePolicyIndexCountOld + movePolicyIndexCount);
        policyIndices.Reserve(cumulativePolicyIndexCountNew);
        policyIndices.AddNAlreadyReserved(movePolicyIndexCount);
        policyValues.Reserve(cumulativePolicyIndexCountNew);
        policyValues.AddNAlreadyReserved(movePolicyIndexCount);
        scratchGame.GeneratePolicyCompressed(game.childVisits[m],
            policyIndices.mutable_data() + cumulativePolicyIndexCountOld,
            policyValues.mutable_data() + cumulativePolicyIndexCountOld);

        scratchGame.ApplyMove(Move(game.moves[m]));
    }
}

// https://www.tensorflow.org/tutorials/load_data/tfrecord
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto
//
// TFRecords format details
//
// A TFRecord file contains a sequence of records. The file can only be read sequentially.
//
// Each record contains a byte-string, for the data-payload, plus the data-length, and CRC32C (32-bit CRC using the Castagnoli polynomial) hashes for integrity checking.
//
// Each record is stored in the following formats:
//
// uint64 length
// uint32 masked_crc32_of_length
// byte   data[length]
// uint32 masked_crc32_of_data
//
// The records are concatenated together to produce the file. CRCs are described here, and the mask of a CRC is:
//
// masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
//
void Storage::WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message)
{
    // Prepare a stream wrapper that can do efficient size-ensuring for writes.
    uint8_t* target;
    google::protobuf::io::EpsCopyOutputStream epsCopy(&stream,
        google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(),
        &target);

    // Serialize the message.
    message.SerializeToString(&buffer);

    // Write the header: length + masked_crc32_of_length
    const uint64_t length = buffer.size();
    const uint32_t lengthCrc = MaskCrc32cForTfRecord(crc32c::Crc32c(reinterpret_cast<const uint8_t*>(&length), sizeof(length)));

    // Just assume we're running on little-endian.
    target = epsCopy.EnsureSpace(target);
    std::memcpy(target, &length, sizeof(length));
    target += sizeof(length);
    std::memcpy(target, &lengthCrc, sizeof(lengthCrc));
    target += sizeof(lengthCrc);

    // Write the payload: data[length]
    target = epsCopy.WriteRaw(buffer.data(), static_cast<int>(buffer.size()), target);

    // Write the footer: masked_crc32_of_data
    const uint32_t dataCrc = MaskCrc32cForTfRecord(crc32c::Crc32c(buffer.data(), buffer.size()));

    // Only writing one datum, same cost as EnsureSpace. Again, just assume we're running on little-endian.
    target = epsCopy.WriteRaw(&dataCrc, sizeof(dataCrc), target);
    epsCopy.Trim(target);
}

uint32_t Storage::MaskCrc32cForTfRecord(uint32_t crc32c)
{
    return ((crc32c >> 15) | (crc32c << 17)) + 0xa282ead8ul;
}

std::filesystem::path Storage::LogPath() const
{
    return _logsPath;
}

std::filesystem::path Storage::MakePath(const std::filesystem::path& root, const std::filesystem::path& path)
{
    // Empty paths have special meaning as N/A.
    if (path.empty())
    {
        return path;
    }

    // Root any relative paths at ChessCoach's appdata directory.
    if (path.is_absolute())
    {
        std::filesystem::create_directories(path);
        return path;
    }

    const std::filesystem::path rooted = (root / path);
    std::filesystem::create_directories(rooted);
    return rooted;
}

void Storage::LoadCommentary()
{
    const Preprocessor preprocessor;
    const GameType gameType = GameType_Supervised; // TODO: Always supervised for now

    _commentary.games.clear();
    _commentary.comments.clear();

    // Load and pre-process commentary, culling empty/unreferenced comments and games.
    for (auto&& entry : std::filesystem::recursive_directory_iterator(_commentaryPaths[gameType]))
    {
        if (entry.path().extension().string() == ".pgn")
        {
            std::ifstream pgnFile = std::ifstream(entry.path(), std::ios::in);
            Pgn::ParsePgn(pgnFile, [&](SavedGame&& game, Commentary&& commentary)
                {
                    int gameIndex = -1;
                    for (auto comment : commentary.comments)
                    {
                        preprocessor.PreprocessComment(comment.comment);
                        if (!comment.comment.empty())
                        {
                            if (gameIndex == -1)
                            {
                                _commentary.games.emplace_back(std::move(game));
                                gameIndex = (static_cast<int>(_commentary.games.size()) - 1);
                            }
                            _commentary.comments.emplace_back(gameIndex, comment.moveIndex, std::move(comment.variationMoves), std::move(comment.comment));
                        }
                    }
                });
        }
    }

    // Generate a vocabulary document with unique comments.
    std::set<std::string> vocabulary;
    for (const SavedComment& comment : _commentary.comments)
    {
        vocabulary.insert(comment.comment);
    }
    const std::filesystem::path vocabularyPath = (_commentaryPaths[gameType] / _vocabularyFilename);
    std::ofstream vocabularyFile = std::ofstream(vocabularyPath, std::ios::out);
    for (const std::string& comment : vocabulary)
    {
        vocabularyFile << comment << std::endl;
    }

    std::cout << "Loaded " << _commentary.comments.size() << " move comments" << std::endl;
}

//CommentaryTrainingBatch* Storage::SampleCommentaryBatch()
//{
//    // Load comments if needed.
//    if (_commentary.comments.empty())
//    {
//        LoadCommentary();
//    }
//
//    // Make sure that there are enough comments to sample from. Just require the batch size for now.
//    if (_commentary.comments.size() < _trainingCommentaryBatchSize)
//    {
//        return nullptr;
//    }
//
//    _commentaryBatch.images.resize(_trainingCommentaryBatchSize);
//    _commentaryBatch.comments.resize(_trainingCommentaryBatchSize);
//
//    std::uniform_int_distribution<int> commentDistribution(0, static_cast<int>(_commentary.comments.size()) - 1);
//
//    for (int i = 0; i < _commentaryBatch.images.size(); i++)
//    {
//        const int commentIndex =
//#if SAMPLE_BATCH_FIXED
//            i;
//#else
//            commentDistribution(Random::Engine);
//#endif
//
//        const SavedComment& comment = _commentary.comments[i];
//        const SavedGame& game = _commentary.games[comment.gameIndex];
//
//        // Find the position for the chosen comment and populate the image and comment text.
//        //
//        // For now interpret the comment as refering to the position after playing the move,
//        // so play moves up to *and including* the stored moveIndex.
//        Game scratchGame = _startingPosition;
//        for (int m = 0; m <= comment.moveIndex; m++)
//        {
//            scratchGame.ApplyMove(Move(game.moves[m]));
//        }
//
//        // Also play out the variation.
//        for (uint16_t move : comment.variationMoves)
//        {
//            scratchGame.ApplyMove(Move(move));
//        }
//
//        scratchGame.GenerateImage(_commentaryBatch.images[i]);
//        _commentaryBatch.comments[i] = comment.comment;
//    }
//
//    return &_commentaryBatch;
//}
