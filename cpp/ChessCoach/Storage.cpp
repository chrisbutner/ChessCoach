#include "Storage.h"

#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <set>

#include "Config.h"
#include "Pgn.h"
#include "Platform.h"
#include "Preprocessing.h"

Storage::Storage(const NetworkConfig& networkConfig, const MiscConfig& miscConfig)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _trainingCommentaryBatchSize(networkConfig.Training.CommentaryBatchSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
    , _vocabularyFilename(networkConfig.Training.VocabularyFilename)
{
    const std::filesystem::path rootPath = Platform::UserDataPath();

    static_assert(GameType_Count == 2);
    _gamesPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.GamesPathTraining);
    _gamesPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.GamesPathValidation);
    _commentaryPaths[GameType_Training] = MakePath(rootPath, networkConfig.Training.CommentaryPathTraining);
    _commentaryPaths[GameType_Validation] = MakePath(rootPath, networkConfig.Training.CommentaryPathValidation);
    _pgnsPath = MakePath(rootPath, miscConfig.Paths_Pgns);
    _networksPath = MakePath(rootPath, miscConfig.Paths_Networks);
    _logsPath = MakePath(rootPath, miscConfig.Paths_Logs);

    InitializePipelines(networkConfig);
}

// Used for testing
Storage::Storage(const NetworkConfig& networkConfig,
    const std::filesystem::path& gamesTrainPath, const std::filesystem::path& gamesValidationPath,
    const std::filesystem::path& pgnsPath, const std::filesystem::path& networksPath)
    : _gameFileCount{}
    , _loadedGameCount{}
    , _currentSaveFile{}
    , _currentSaveGameCount{}
    , _trainingBatchSize(networkConfig.Training.BatchSize)
    , _trainingCommentaryBatchSize(networkConfig.Training.CommentaryBatchSize)
    , _pgnInterval(networkConfig.Training.PgnInterval)
    , _vocabularyFilename() // TODO: Update with commentary unit tests
    , _gamesPaths{ gamesTrainPath, gamesValidationPath }
    , _commentaryPaths { "", "" } // TODO: Update with commentary unit tests
    , _pgnsPath(pgnsPath)
    , _networksPath(networksPath)
{
    static_assert(GameType_Count == 2);

    InitializePipelines(networkConfig);
}

void Storage::InitializePipelines(const NetworkConfig& networkConfig)
{
    static_assert(GameType_Count == 2);
    _pipelines[GameType_Training].Initialize(_games[GameType_Training], networkConfig.Training.BatchSize, 2 /* workerCount */);
    _pipelines[GameType_Validation].Initialize(_games[GameType_Validation], networkConfig.Training.BatchSize, 2 /* workerCount */);
}

void Storage::LoadExistingGames(GameType gameType, int maxLoadCount)
{
    // Should be no contention at initialization time. Just bulk lock around
    // (a) AddGameWithoutSaving, and (b) setting _gameFileCount/_loadedGameCount.
    std::lock_guard lock(_mutex);

    if (_gamesPaths[gameType].empty())
    {
        return;
    }

    ReplayBuffer& games = _games[gameType];
    const int gameFileCount = static_cast<int>(std::distance(std::filesystem::directory_iterator(_gamesPaths[gameType]), std::filesystem::directory_iterator()));

    int lastPrintedCount = 0;
    int loadedCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
    {
        const int maxAdditionalCount = (maxLoadCount - loadedCount);
        const int justLoadedCount = LoadFromDiskInternal(entry.path(),
            std::bind(&ReplayBuffer::AddGame, &games, std::placeholders::_1),
            maxAdditionalCount);
        loadedCount += justLoadedCount;

        const int printCount = (loadedCount - (loadedCount % 1000));
        if (printCount > lastPrintedCount)
        {
            std::cout << printCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
            lastPrintedCount = printCount;
        }
        if (loadedCount >= maxLoadCount)
        {
            break;
        }
    }

    if (loadedCount > lastPrintedCount)
    {
        std::cout << loadedCount << " games loaded (" << GameTypeNames[gameType] << ")" << std::endl;
        lastPrintedCount = loadedCount;
    }

    _gameFileCount[gameType] = gameFileCount;
    _loadedGameCount[gameType] = loadedCount;
}

int Storage::AddGame(GameType gameType, SavedGame&& game)
{
    std::lock_guard lock(_mutex);
    
    const SavedGame& emplaced = _games[gameType].AddGame(std::move(game));

    SaveToDisk(gameType, emplaced);

    const int gameNumber = ++_loadedGameCount[gameType];
    
    if ((gameNumber % _pgnInterval) == 0)
    {
        std::stringstream suffix;
        suffix << std::setfill('0') << std::setw(9) << gameNumber;
        const std::string& filename = ("game_" + suffix.str() + ".pgn");
        const std::filesystem::path pgnPath = _pgnsPath / filename;

        std::ofstream pgnFile = std::ofstream(pgnPath, std::ios::out);
        Pgn::GeneratePgn(pgnFile, emplaced);
    }

    return gameNumber;
}

// No locking: assumes single-threaded sampling periods without games being played.
TrainingBatch* Storage::SampleBatch(GameType gameType)
{
    return _pipelines[gameType].SampleBatch();
}

int Storage::GamesPlayed(GameType gameType) const
{
    std::lock_guard lock(_mutex);

    return _loadedGameCount[gameType];
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

// Requires caller to lock.
void Storage::SaveToDisk(GameType gameType, const SavedGame& game)
{
    // Check whether the current save file has run out of room.
    if (_currentSaveFile[gameType].is_open())
    {
        if (_currentSaveGameCount[gameType] >= Config::Misc.Storage_MaxGamesPerFile)
        {
            _currentSaveFile[gameType] = {};
            _currentSaveGameCount[gameType] = 0;
        }
    }
    // Try to continue from a previous run, but only add to the final file, not any "inside" gaps.
    // This should only run once; future SaveToDisk calls should always see an open _currentSaveFile[gameType].
    else
    {
        std::filesystem::directory_entry lastEntry;
        for (const auto& entry : std::filesystem::directory_iterator(_gamesPaths[gameType]))
        {
            lastEntry = entry;
        }

        std::ifstream lastFile;
        if (!lastEntry.path().empty() && (lastFile = std::ifstream(lastEntry.path(), std::ios::in | std::ios::binary)))
        {
            Version version;
            GameCount gameCount;

            lastFile.read(reinterpret_cast<char*>(&version), sizeof(version));
            lastFile.read(reinterpret_cast<char*>(&gameCount), sizeof(gameCount));
            assert(version == Version1);

            if (gameCount < Config::Misc.Storage_MaxGamesPerFile)
            {
                lastFile.close();
                _currentSaveFile[gameType] = std::ofstream(lastEntry.path(), std::ios::in | std::ios::out | std::ios::binary);
                _currentSaveGameCount[gameType] = gameCount;
            }
        }
    }

    // We may need to start a new file.
    if (!_currentSaveFile[gameType].is_open())
    {
        const int newGameFileNumber = ++_gameFileCount[gameType];
        std::filesystem::path gamesPath = _gamesPaths[gameType] / GenerateGamesFilename(newGameFileNumber);
        _currentSaveFile[gameType] = std::ofstream(gamesPath, std::ios::out | std::ios::binary);
        _currentSaveGameCount[gameType] = 0;

        const Version version = Version1;
        const GameCount initialGameCount = 0;
        _currentSaveFile[gameType].write(reinterpret_cast<const char*>(&version), sizeof(version));
        _currentSaveFile[gameType].write(reinterpret_cast<const char*>(&initialGameCount), sizeof(initialGameCount));
    }

    // Save the game and flush to disk.
    const GameCount newGameCountInFile = ++_currentSaveGameCount[gameType];
    SaveToDiskInternal(_currentSaveFile[gameType], game, newGameCountInFile);
    _currentSaveFile[gameType].flush();
}

int Storage::LoadFromDiskInternal(const std::filesystem::path& path, std::function<void(SavedGame&&)> gameHandler, int maxLoadCount)
{
    std::ifstream file = std::ifstream(path, std::ios::in | std::ios::binary);

    Version version;
    GameCount gameCount;

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&gameCount), sizeof(gameCount));
    assert(version == Version1);

    const int loadCount = std::min(maxLoadCount, static_cast<int>(gameCount));
    for (int i = 0; i < loadCount; i++)
    {
        MoveCount moveCount;
        float result;

        file.read(reinterpret_cast<char*>(&moveCount), sizeof(moveCount));
        file.read(reinterpret_cast<char*>(&result), sizeof(result));
        assert(moveCount >= 1);
        assert((result == CHESSCOACH_VALUE_WIN) ||
            (result == CHESSCOACH_VALUE_DRAW) ||
            (result == CHESSCOACH_VALUE_LOSS));

        std::vector<uint16_t> moves(moveCount);
        file.read(reinterpret_cast<char*>(moves.data()), sizeof(uint16_t) * moveCount);

        std::vector<float> mctsValues(moveCount);
        file.read(reinterpret_cast<char*>(mctsValues.data()), sizeof(float) * moveCount);

        std::vector<std::map<Move, float>> childVisits(moveCount);
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

        gameHandler(SavedGame(result, std::move(moves), std::move(mctsValues), std::move(childVisits)));
    }

    return loadCount;
}

void Storage::SaveMultipleGamesToDisk(const std::filesystem::path& path, const std::vector<SavedGame>& games)
{
    std::ofstream file = std::ofstream(path, std::ios::out | std::ios::binary);

    const Version version = Version1;
    const GameCount gameCount = static_cast<GameCount>(games.size());
    assert(games.size() <= std::numeric_limits<GameCount>::max());

    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&gameCount), sizeof(gameCount));

    for (const SavedGame& game : games)
    {
        SaveToDiskInternal(file, game);
    }
}

void Storage::SaveToDiskInternal(std::ofstream& file, const SavedGame& game, GameCount newGameCountInFile)
{
    assert(newGameCountInFile <= std::numeric_limits<GameCount>::max());

    // Update the game count.
    file.seekp(sizeof(Version), file.beg);
    file.write(reinterpret_cast<const char*>(&newGameCountInFile), sizeof(newGameCountInFile));
    file.seekp(0, file.end);

    // Write the game.
    SaveToDiskInternal(file, game);
}


void Storage::SaveToDiskInternal(std::ofstream& file, const SavedGame& game)
{
    // Write the game.
    const MoveCount moveCount = static_cast<MoveCount>(game.moves.size());

    file.write(reinterpret_cast<const char*>(&moveCount), sizeof(moveCount));
    file.write(reinterpret_cast<const char*>(&game.result), sizeof(game.result));

    file.write(reinterpret_cast<const char*>(game.moves.data()), sizeof(uint16_t) * moveCount);

    file.write(reinterpret_cast<const char*>(game.mctsValues.data()), sizeof(float) * moveCount);

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

std::string Storage::GenerateGamesFilename(int gamesNumber)
{
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(9) << gamesNumber;

    return "games_" + suffix.str();
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
    const GameType gameType = GameType_Training; // TODO: Always training for now

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

Window Storage::GetWindow(GameType gameType) const
{
    return _games[gameType].GetWindow();
}

void Storage::SetWindow(GameType gameType, const Window& window)
{
    std::cout << "Setting window: " << window.TrainingGameMin << " -> " << window.TrainingGameMax
        << " (" << GameTypeNames[gameType] << ")" << std::endl;
    _games[gameType].SetWindow(window);
}

CommentaryTrainingBatch* Storage::SampleCommentaryBatch()
{
    // Load comments if needed.
    if (_commentary.comments.empty())
    {
        LoadCommentary();
    }

    // Make sure that there are enough comments to sample from. Just require the batch size for now.
    if (_commentary.comments.size() < _trainingCommentaryBatchSize)
    {
        return nullptr;
    }

    _commentaryBatch.images.resize(_trainingCommentaryBatchSize);
    _commentaryBatch.comments.resize(_trainingCommentaryBatchSize);

    std::uniform_int_distribution<int> commentDistribution(0, static_cast<int>(_commentary.comments.size()) - 1);

    for (int i = 0; i < _commentaryBatch.images.size(); i++)
    {
        const int commentIndex =
#if SAMPLE_BATCH_FIXED
            i;
#else
            commentDistribution(Random::Engine);
#endif

        const SavedComment& comment = _commentary.comments[i];
        const SavedGame& game = _commentary.games[comment.gameIndex];

        // Find the position for the chosen comment and populate the image and comment text.
        //
        // For now interpret the comment as refering to the position after playing the move,
        // so play moves up to *and including* the stored moveIndex.
        Game scratchGame = _startingPosition;
        for (int m = 0; m <= comment.moveIndex; m++)
        {
            scratchGame.ApplyMove(Move(game.moves[m]));
        }

        // Also play out the variation.
        for (uint16_t move : comment.variationMoves)
        {
            scratchGame.ApplyMove(Move(move));
        }

        _commentaryBatch.images[i] = scratchGame.GenerateImage();
        _commentaryBatch.comments[i] = comment.comment;
    }

    return &_commentaryBatch;
}