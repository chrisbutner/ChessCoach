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

#include "Storage.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>
#include <ctime>

#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#pragma warning(disable:4100) // Ignore unused args in generated code
#pragma warning(disable:4127) // Ignore const-per-architecture warning
#include <protobuf/ChessCoach.pb.h>
#pragma warning(disable:4127) // Ignore const-per-architecture warning
#pragma warning(default:4100) // Ignore unused args in generated code

#include <crc32c/include/crc32c/crc32c.h>
#include <Stockfish/movegen.h>

#include "Config.h"
#include "Pgn.h"
#include "Platform.h"
#include "Preprocessing.h"
#include "Random.h"

Storage::Storage()
    : _trainingGameCount(0)
    , _gamesPerChunk(Config::Misc.Storage_GamesPerChunk)
    , _pgnInterval(Config::Network.Training.PgnInterval)
    , _sessionNonce("UNINITIALIZED")
    , _sessionGameCount(0)
    , _sessionChunkCount(0)
{
    _relativeTrainingGamePath = Config::Network.Training.GamesPathTraining;
    _localTrainingGamePath = MakeLocalPath(_relativeTrainingGamePath);
    _relativePgnsPath = Config::Misc.Paths_Pgns;
}

void Storage::InitializeLocalGamesChunks(INetwork* network)
{
    // Use a 32-bit session nonce to help differentiate this run from others. Still secondary to timestamp in ordering.
    const std::string alphabet = "0123456789ABCDEF";
    std::uniform_int_distribution<> distribution(0, static_cast<int>(alphabet.size()) - 1);
    _sessionNonce = std::string(8, '\0');
    for (char& c : _sessionNonce)
    {
        c = alphabet[distribution(Random::Engine)];
    }

    // Count training games previously played and saved locally without yet being chunked.
    _trainingGameCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(_localTrainingGamePath))
    {
        if (entry.path().extension().string() == ".game")
        {
            _trainingGameCount++;
        }
    }

    // Try to chunk now in case we already have enough games (so zero would be played)
    // but they failed to chunk previously.
    if (_trainingGameCount >= _gamesPerChunk)
    {
        TryChunkMultiple(network);
    }
}

// AddTrainingGame can be called from multiple self-play worker threads.
int Storage::AddTrainingGame(INetwork* network, SavedGame&& game)
{
    // Give this game a number and filename.
    const int gameNumber = ++_sessionGameCount;
    const std::string filenameStem = GenerateFilename(gameNumber);

    // Save locally for chunking later.
    const std::filesystem::path localGamePath = _localTrainingGamePath / (filenameStem + ".game");
    SaveChunk(localGamePath, { game });

    // Occasionally save PGNs to central storage.
    if ((gameNumber % _pgnInterval) == 0)
    {
        const std::filesystem::path relativePgnPath = _relativePgnsPath / (filenameStem + ".pgn");

        std::stringstream buffer;
        Pgn::GeneratePgn(buffer, game);

        network->SaveFile(relativePgnPath.string(), buffer.str());
    }

    // When enough individual games have been saved, chunk and store centrally.
    // Use the atomic_int to ensure that only one caller attempts to chunk, and
    // assume that _gamesPerChunk is large enough that the chunking will finish
    // well before it is time for the next one.
    const int newTrainingGameCount = ++_trainingGameCount;
    if ((newTrainingGameCount % _gamesPerChunk) == 0)
    {
        TryChunkMultiple(network);
    }

    return gameNumber;
}

// Just in case anything went wrong with file I/O, etc. previously, attempt to
// create as many chunks as we can here. This is still safe with the outer atomic_int
// check as long as we finish before the next _gamesPerChunk cycle.
void Storage::TryChunkMultiple(INetwork* network)
{
    std::vector<std::filesystem::path> gamePaths;
    for (const auto& entry : std::filesystem::directory_iterator(_localTrainingGamePath))
    {
        if (entry.path().extension().string() == ".game")
        {
            gamePaths.emplace_back(entry.path());
        }
        if (gamePaths.size() == _gamesPerChunk)
        {
            ChunkGames(network, gamePaths);
            gamePaths.clear();
        }
    }
}

void Storage::ChunkGames(INetwork* network, std::vector<std::filesystem::path>& gamePaths)
{
    // Set up a buffer for the TFRecord file contents, compressing with zlib.
    // Reserve 128 MB in advance, roughly enough to hold any chunk.
    std::string buffer;
    {
        buffer.reserve(128 * 1024 * 1024);
        google::protobuf::io::StringOutputStream chunkWrapped(&buffer);
        google::protobuf::io::GzipOutputStream::Options zipOptions{};
        zipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
        google::protobuf::io::GzipOutputStream chunkZip(&chunkWrapped, zipOptions);

        // Just decompress each individual game chunk with zlib and append to the chunk.
        for (auto& path : gamePaths)
        {
            PosixFile gameFile(path, false /* write */);
            google::protobuf::io::FileInputStream gameWrapped(gameFile.FileDescriptor());
            google::protobuf::io::GzipInputStream gameZip(&gameWrapped, google::protobuf::io::GzipInputStream::ZLIB);

            void* chunkBuffer;
            const void* gameBuffer;
            int chunkSize;
            int gameSize;

            // Ordering is important here: check short-lived "gameZip" first (is there data to read)
            // then long-lived "chunkZip" (can we write it) so that "BackUp" is never missed.
            while (gameZip.Next(&gameBuffer, &gameSize) && chunkZip.Next(&chunkBuffer, &chunkSize))
            {
                const int copySize = std::min(gameSize, chunkSize);
                std::memcpy(chunkBuffer, gameBuffer, copySize);

                chunkZip.BackUp(chunkSize - copySize);
                gameZip.BackUp(gameSize - copySize);
            }
        }
    }

    // Write the chunk to central storage.
    const int chunkNumber = ++_sessionChunkCount;
    const std::string& filename = GenerateFilename(chunkNumber) + ".chunk";
    const auto& relativePath = (_relativeTrainingGamePath / filename);
    std::cout << "Chunking " << _gamesPerChunk << " games to " << filename << std::endl;
    network->SaveFile(relativePath.string(), buffer);

    // Delete the individual games.
    for (auto& path : gamePaths)
    {
        std::filesystem::remove(path);
    }

    // Update stats.
    _trainingGameCount -= _gamesPerChunk;
}

// Training is only done on chunks, not individual games, so round the target up to the nearest chunk.
int Storage::TrainingGamesToPlay(int trainingChunkCount, int targetGameCount, bool ignoreLocalGames) const
{
    const int localGameCount = (ignoreLocalGames ? 0 : static_cast<int>(_trainingGameCount));
    const int existingCount = ((trainingChunkCount * _gamesPerChunk) + localGameCount);
    const int roundedTarget = (((targetGameCount + _gamesPerChunk - 1) / _gamesPerChunk) * _gamesPerChunk);
    return std::max(0, roundedTarget - existingCount);
}

std::string Storage::GenerateSimpleChunkFilename(int chunkNumber) const
{
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << chunkNumber << ".chunk";
    return suffix.str();
}

std::string Storage::GenerateFilename(int number)
{
    std::stringstream filename;

    // Format as "YYmmdd_HHMMSS_milliseconds_sessionnonce_number"; e.g. "20201022_181546_008_24FFE8F502A72C8D_000000005".
    // Use UCT rather than local time for comparability across multiple machines, including a local/cloud mix.
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    filename << std::put_time(std::gmtime(&time), "%Y%m%d_%H%M%S");
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    filename << "_" << std::setfill('0') << std::setw(3) << milliseconds << "_" << _sessionNonce << "_"
        << std::setfill('0') << std::setw(9) << number;
    
    return filename.str();
}

void Storage::SaveChunk(const std::filesystem::path& path, const std::vector<SavedGame>& games) const
{
    // Compress the TFRecord file using zlib.
    PosixFile file(path, true /* write */);
    google::protobuf::io::FileOutputStream wrapped(file.FileDescriptor());
    google::protobuf::io::GzipOutputStream::Options zipOptions{};
    zipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
    google::protobuf::io::GzipOutputStream zip(&wrapped, zipOptions);

    // Write a "tf.train.Example" protobuf for each game as a TFRecord.
    message::Example storeGame;
    std::string buffer;
    for (const SavedGame& game : games)
    {
        PopulateGame(_startingPosition, game, storeGame);
        WriteTfRecord(zip, buffer, storeGame);
    }
}

// This is only used to drive the ChessCoachGui tool. TFRecords are normally loaded
// using the tf.data pipeline in dataset.py but that's 100x-1000x too slow for
// random access to games and positions.
void Storage::LoadGameFromChunk(const std::string& chunkContents, int gameIndex, SavedGame* gameOut)
{
    google::protobuf::io::ArrayInputStream wrapped(chunkContents.data(), static_cast<int>(chunkContents.size()));
    google::protobuf::io::GzipInputStream zip(&wrapped, google::protobuf::io::GzipInputStream::ZLIB);

    // The chunk is stored as a bunch of concatenated TFRecords, using zlib.
    for (int i = 0; i < gameIndex; i++)
    {
        if (!SkipTfRecord(zip))
        {
            throw ChessCoachException("Failed to parse chunk");
        }
    }

    // Read the payload length and skip its crc32c.
    uint64_t payloadLength;
    if (!Read(zip, payloadLength) || !zip.Skip(sizeof(uint32_t)))
    {
        throw ChessCoachException("Failed to parse chunk");
    }

    // The game is stored as a TFRecord using zlib.
    message::Example compressedGame;
    if (!compressedGame.MergePartialFromBoundedZeroCopyStream(&zip, static_cast<int>(payloadLength)))
    {
        throw ChessCoachException("Failed to parse game");
    }

    // The first 12 planes of "image_pieces_auxiliary" contain the pieces for each position in the game.
    auto& features = compressedGame.features().feature();
    auto& result = features.at("result").float_list().value();
    auto& mctsValues = features.at("mcts_values").float_list().value();
    auto& imagePiecesAuxiliary = features.at("image_pieces_auxiliary").int64_list().value();
    const int imagePiecesAuxiliaryStride = (INetwork::InputPieceAndRepetitionPlanesPerPosition + INetwork::InputAuxiliaryPlaneCount);
    auto& policyRowLengths = features.at("policy_row_lengths").int64_list().value();
    auto& policyIndices = features.at("policy_indices").int64_list().value();
    auto& policyValues = features.at("policy_values").float_list().value();

    // Set up result and MCTS values directly.
    // MCTS deals with probabilities in [0, 1]. Network deals with tanh outputs/targets in (-1, 1)/[-1, 1].
    *gameOut = SavedGame();
    gameOut->result = INetwork::MapProbability11To01(result[0]);
    gameOut->moveCount = mctsValues.size();
    gameOut->mctsValues.insert(gameOut->mctsValues.begin(), mctsValues.begin(), mctsValues.end());
    INetwork::MapProbabilities11To01(gameOut->mctsValues.size(), gameOut->mctsValues.data());

    // Play out the game and match the resulting pieces after each legal move.
    Game game;
    int policyStart = 0;
    for (int m = 0; m < gameOut->moveCount; m++)
    {
        // Reconstruct the policy by walking over legal moves.
        const int policyLength = static_cast<int>(policyRowLengths[m]);
        const int64_t* positionPolicyIndices = (policyIndices.data() + policyStart);
        const float* positionPolicyValues = (policyValues.data() + policyStart);
        policyStart += policyLength;

        std::unique_ptr<INetwork::OutputPlanes> policy = std::make_unique<INetwork::OutputPlanes>(); // Zero for "GeneratePolicyDecompress"
        game.GeneratePolicyDecompress(policyLength, positionPolicyIndices, positionPolicyValues, *policy);

        std::map<Move, float>& childVisits = gameOut->childVisits.emplace_back();
        const MoveList legalMoves = MoveList<LEGAL>(game.GetPosition());
        for (const Move move : legalMoves)
        {
            childVisits[move] = game.PolicyValue(*policy, move);
        }

        // We can't find the final move by matching pieces since the terminal position is left off,
        // so guess instead using the game result and the policy for the final position.
        const int resultingPosition = (m + 1);
        if (resultingPosition < gameOut->moveCount)
        {
            const Move move = game.ApplyMoveInfer(reinterpret_cast<const INetwork::PackedPlane*>(
                imagePiecesAuxiliary.data()) + (resultingPosition * imagePiecesAuxiliaryStride));
            gameOut->moves.push_back(static_cast<uint16_t>(move));
        }
        else
        {
            const Move move = game.ApplyMoveGuess(gameOut->result, childVisits);
            gameOut->moves.push_back(static_cast<uint16_t>(move));
        }
    }
}

bool Storage::SkipTfRecord(google::protobuf::io::ZeroCopyInputStream& stream) const
{
    uint64_t payloadLength;
    bool ok = Read(stream, payloadLength);
    if (ok)
    {
        // Skip the two CRC32Cs plus the payload.
        ok = stream.Skip(static_cast<int>(payloadLength) + (2 * sizeof(uint32_t)));
    }
    return ok;
}

#pragma warning(disable:4706) // Intentionally assigning, not comparing
template <typename T>
bool Storage::Read(google::protobuf::io::ZeroCopyInputStream& stream, T& value) const
{
    bool ok;
    const void* buffer;
    int size = 0;
    int offset = 0;
    while ((ok = stream.Next(&buffer, &size)))
    {
        const int copyLength = std::min(size, static_cast<int>(sizeof(value)) - offset);
        if (copyLength > 0)
        {
            // Just assume we're running on little-endian.
            std::memcpy(reinterpret_cast<char*>(&value) + offset, buffer, copyLength);
            stream.BackUp(size - copyLength);

            offset += copyLength;
            if (offset >= sizeof(value))
            {
                break;
            }
        }
    }
    return ok;
}
#pragma warning(default:4706) // Intentionally assigning, not comparing
template bool Storage::Read<int>(google::protobuf::io::ZeroCopyInputStream& stream, int& value) const;
template bool Storage::Read<uint64_t>(google::protobuf::io::ZeroCopyInputStream& stream, uint64_t& value) const;

message::Example Storage::DebugPopulateGame(const SavedGame& game) const
{
    message::Example gameOut;
    PopulateGame(_startingPosition, game, gameOut);
    return gameOut;
}

void Storage::PopulateGame(Game scratchGame, const SavedGame& game, message::Example& gameOut) const
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

    // Fix up result and MCTS values.
    // MCTS deals with probabilities in [0, 1]. Network deals with tanh outputs/targets in (-1, 1)/[-1, 1].
    INetwork::MapProbabilities01To11(result.size(), result.mutable_data());
    INetwork::MapProbabilities01To11(mctsValues.size(), mctsValues.mutable_data());

    // Image and policy require applying moves to a scratch game, so process a move-at-once.
    // Policy indices/values are ragged, so reserve for each move.
    auto& imagePiecesAuxiliary = *features["image_pieces_auxiliary"].mutable_int64_list()->mutable_value();
    imagePiecesAuxiliary.Clear();
    const int imagePiecesAuxiliaryStride = (INetwork::InputPieceAndRepetitionPlanesPerPosition + INetwork::InputAuxiliaryPlaneCount);
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
        INetwork::PackedPlane* imageAuxiliaryOut = (imagePiecesOut + INetwork::InputPieceAndRepetitionPlanesPerPosition);
        scratchGame.GenerateImageCompressed(imagePiecesOut, imageAuxiliaryOut);

        const int movePolicyIndexCount = static_cast<int>(game.childVisits[m].size());
        policyRowLengths[m] = movePolicyIndexCount;

        const int cumulativePolicyIndexCountOld = policyIndices.size();
        const int cumulativePolicyIndexCountNew = (cumulativePolicyIndexCountOld + movePolicyIndexCount);
        policyIndices.Reserve(cumulativePolicyIndexCountNew);
        policyIndices.AddNAlreadyReserved(movePolicyIndexCount);
        policyValues.Reserve(cumulativePolicyIndexCountNew); // Policy values get zeroed here, as required by "GeneratePolicyCompressed".
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
void Storage::WriteTfRecord(google::protobuf::io::ZeroCopyOutputStream& stream, std::string& buffer, const google::protobuf::Message& message) const
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

uint32_t Storage::MaskCrc32cForTfRecord(uint32_t crc32c) const
{
    return ((crc32c >> 15) | (crc32c << 17)) + 0xa282ead8ul;
}

// We can only write to local paths from C++ and need to call in to Python to write to gs:// locations
// when running on Google Cloud with TPUs. That's okay for local gathering/chunking of games, and UCI logging.
std::filesystem::path Storage::MakeLocalPath(const std::filesystem::path& path)
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

    const std::filesystem::path root = Platform::UserDataPath();
    const std::filesystem::path rooted = (root / path);
    std::filesystem::create_directories(rooted);
    return rooted;
}

void Storage::SaveCommentary(CommentarySaveContext& saveContext, const std::vector<SavedGame>& games,
    std::vector<SavedCommentary>& gameCommentary, Vocabulary& vocabulary) const
{
    // Only let one thread contribute to the shared save context at once.
    std::lock_guard lock(saveContext.mutex);

    const Preprocessor preprocessor;

    for (int i = 0; i < games.size(); i++)
    {
        const SavedGame& game = games[i];
        SavedCommentary commentary = gameCommentary[i];

        // Require that commentary for this game refers to positions in order
        // and play out a single base scratch game, then branch off variations.
        //
        // Variations "override" the last real move and so will regress the move index,
        // so sort comments by move index here.
        Game scratchGame = _startingPosition;
        std::sort(commentary.comments.begin(), commentary.comments.end(), [](const auto& a, const auto& b) {
            return (a.moveIndex < b.moveIndex);
        });

        for (SavedComment& comment : commentary.comments)
        {
            preprocessor.PreprocessComment(comment.comment);
            if (!comment.comment.empty())
            {
                // Write a single "tf.train.Example" protobuf for all images/comments as a TFRecord.
                // Choose whether to add to training or validation.
                CommentaryRecordType& recordType = saveContext.ChooseRecordType();
                auto& features = *recordType.record->mutable_features()->mutable_feature();
                auto& images = *features["images"].mutable_int64_list()->mutable_value();
                auto& comments = *features["comments"].mutable_bytes_list()->mutable_value();
                const int commentaryImageStride = INetwork::CommentaryInputPlaneCount;

                // Update vocabulary.
                vocabulary.vocabulary.push_back(comment.comment);

                // Write the comment directly (don't touch "comment.comment" after this).
                comments.Add(std::move(comment.comment));

                // Prepare the image for writing.
                const int imageSizeOld = images.size();
                const int imageSizeNew = (imageSizeOld + commentaryImageStride);
                images.Reserve(imageSizeNew);
                images.AddNAlreadyReserved(commentaryImageStride);
                INetwork::PackedPlane* imageOut = (reinterpret_cast<INetwork::PackedPlane*>(images.mutable_data()) + imageSizeOld);

                // Find the position for the chosen comment and populate the image.
                //
                // For now interpret the comment as refering to the position after playing the move,
                // so play moves up to *and including* the stored moveIndex.
                for (int m = scratchGame.Ply(); m <= comment.moveIndex; m++)
                {
                    // Some commentary games include null moves in the actual game, not just variations
                    // (e.g. at the very end to add a summary comment) so allow them here too.
                    scratchGame.ApplyMoveMaybeNull(Move(game.moves[m]));
                }

                // Also play out the variation.
                Game variation = scratchGame;
                for (uint16_t move : comment.variationMoves)
                {
                    variation.ApplyMoveMaybeNull(Move(move));
                }

                // Generate the full image: no compression for commentary because of branching variation structure.
                variation.GenerateCommentaryImage(imageOut);

                // Write the record if full.
                if (comments.size() >= CommentarySaveContext::PositionsPerRecord)
                {
                    WriteCommentary(recordType);
                }
            }
        }
    }
}

void Storage::WriteRemainingCommentary(CommentarySaveContext& saveContext) const
{
    // Should only be one caller, but fence vs. most recent other.
    std::lock_guard lock(saveContext.mutex);

    for (CommentaryRecordType& recordType : saveContext.recordTypes)
    {
        if (recordType.record->has_features())
        {
            WriteCommentary(recordType);
        }
    }
}

void Storage::WriteCommentary(CommentaryRecordType& recordType) const
{
    // Generate a path.
    const std::filesystem::path path = (recordType.directory / GenerateSimpleChunkFilename(++recordType.latestRecordNumber));

    // Compress the TFRecord file using zlib.
    PosixFile file(path, true /* write */);
    google::protobuf::io::FileOutputStream wrapped(file.FileDescriptor());
    google::protobuf::io::GzipOutputStream::Options zipOptions{};
    zipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
    google::protobuf::io::GzipOutputStream zip(&wrapped, zipOptions);

    // Write the "tf.train.Example" protobuf.
    std::string buffer;
    WriteTfRecord(zip, buffer, *recordType.record);

    // Clear the record, ready to build up again.
    recordType.record->Clear();
}

CommentaryRecordType& CommentarySaveContext::ChooseRecordType()
{
    std::discrete_distribution distribution(recordWeights.begin(), recordWeights.end());
    return recordTypes[distribution(Random::Engine)];
}

#include <map>
void Storage::FixChunk(INetwork* network, const std::string& filename)
{
    const std::string chunkContents = network->LoadFile(filename);
    google::protobuf::io::ArrayInputStream wrapped(chunkContents.data(), static_cast<int>(chunkContents.size()));
    google::protobuf::io::GzipInputStream zip(&wrapped, google::protobuf::io::GzipInputStream::ZLIB);

    std::string buffer;
    {
        buffer.reserve(128 * 1024 * 1024);
        google::protobuf::io::StringOutputStream chunkWrapped(&buffer);
        google::protobuf::io::GzipOutputStream::Options zipOptions{};
        zipOptions.format = google::protobuf::io::GzipOutputStream::ZLIB;
        google::protobuf::io::GzipOutputStream chunkZip(&chunkWrapped, zipOptions);
        std::string intermediateBuffer;

        // The chunk is stored as a bunch of concatenated TFRecords, using zlib.
        for (int i = 0; i < Config::Misc.Storage_GamesPerChunk; i++)
        {
            // Read the payload length and skip its crc32c.
            uint64_t payloadLength;
            if (!Read(zip, payloadLength) || !zip.Skip(sizeof(uint32_t)))
            {
                throw ChessCoachException("Failed to parse chunk");
            }

            // The game is stored as a TFRecord using zlib.
            message::Example compressedGame;
            if (!compressedGame.MergePartialFromBoundedZeroCopyStream(&zip, static_cast<int>(payloadLength)))
            {
                throw ChessCoachException("Failed to parse game");
            }

            // Skip the payload's crc32c.
            if (!zip.Skip(sizeof(uint32_t)))
            {
                throw ChessCoachException("Failed to parse chunk");
            }

            // The first 12 planes of "image_pieces_auxiliary" contain the pieces for each position in the game.
            auto& features = *compressedGame.mutable_features()->mutable_feature();
            auto& imagePiecesAuxiliary = features.at("image_pieces_auxiliary").int64_list().value();
            const int imagePiecesAuxiliaryStride = (INetwork::InputPieceAndRepetitionPlanesPerPosition + INetwork::InputAuxiliaryPlaneCount);
            auto& policyRowLengths = features.at("policy_row_lengths").int64_list().value();
            auto& policyIndices = *features["policy_indices"].mutable_int64_list()->mutable_value();

            // Play out the game and match the resulting pieces after each legal move.
            Game game;
            int policyStart = 0;
            const int moveCount = policyRowLengths.size();
            for (int m = 0; m < moveCount; m++)
            {
                // Index into the compressed policy by walking over legal moves.
                const int policyLength = static_cast<int>(policyRowLengths[m]);
                int64_t* positionPolicyIndices = (policyIndices.mutable_data() + policyStart);
                policyStart += policyLength;

                // Replace bad queen/knight planes with good ones.
                const MoveList legalMoves = MoveList<LEGAL>(game.GetPosition());
                std::set<int> badIndices;
                std::set<int> goodIndices;
                std::map<int, int> updates;
                for (const Move move : legalMoves)
                {
                    const int badIndex = game.BadPolicyIndex(move);
                    if (badIndices.find(badIndex) != badIndices.end())
                    {
                        throw ChessCoachException("Duplicate bad policy index");
                    }
                    badIndices.insert(badIndex);
                    for (int p = 0; p < policyLength; p++)
                    {
                        if (positionPolicyIndices[p] == badIndex)
                        {
                            const INetwork::PlanesPointerFlat zero = 0;
                            const int goodIndex = static_cast<int>(&game.PolicyValue(zero, move) - zero);
                            if (goodIndices.find(goodIndex) != goodIndices.end())
                            {
                                throw ChessCoachException("Duplicate good policy index");
                            }
                            goodIndices.insert(goodIndex);
                            updates[p] = goodIndex;
                            break;
                        }
                    }
                }
                if ((badIndices.size() != policyLength) ||
                    (goodIndices.size() != policyLength))
                {
                    throw ChessCoachException("Unmatched policy indices");
                }
                for (auto& [index, value] : updates)
                {
                    positionPolicyIndices[index] = value;
                }

                // Use the piece and repetition planes to infer the move played.
                const int resultingPosition = (m + 1);
                if (resultingPosition < moveCount)
                {
                    game.ApplyMoveInfer(reinterpret_cast<const INetwork::PackedPlane*>(
                        imagePiecesAuxiliary.data()) + (resultingPosition * imagePiecesAuxiliaryStride));
                }
            }

            intermediateBuffer.clear();
            WriteTfRecord(chunkZip, intermediateBuffer, compressedGame);
        }
    }

    std::string newFilename = filename;
    newFilename.replace(filename.find("Fresh7a"), 7, "Fresh7b");
    network->SaveFile(newFilename, buffer);
}