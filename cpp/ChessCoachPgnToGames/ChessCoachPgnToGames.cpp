#include <functional>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <filesystem>
#include <atomic>
#include <deque>

#include <Stockfish/position.h>
#include <tclap/CmdLine.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Storage.h>
#include <ChessCoach/Pgn.h>

// Custom binary format: ~15k (MSVC/Win), ~71k (GCC/Linux) games per second on i7-6700, Samsung SSD 950 PRO 512GB.
// Compressed protobuf/planes: ~7.8k (MSVC/Win), ~11.5k (GCC/Linux) games per second on i7-6700, Samsung SSD 950 PRO 512GB.
// Could probably be trivially improved by lessening PGN queue mutex contention.
// Haven't investigated platform/compiler differences, probably easy gains.
class ChessCoachPgnToGames : public ChessCoach
{
public:

    ChessCoachPgnToGames(const std::filesystem::path& inputDirectory, const std::filesystem::path& outputDirectory,
        int threadCount, bool commentary);

    void InitializeLight();
    void FinalizeLight();
    void ConvertAll();

private:

    void ConvertPgns();
    void SaveChunk(const Storage& storage, std::vector<SavedGame>& games,
        std::vector<SavedCommentary>& gameCommentary, Vocabulary& vocabulary);

private:

    std::filesystem::path _inputDirectory;
    std::filesystem::path _outputDirectory;
    int _threadCount;
    bool _commentary;

    std::mutex _pgnQueueMutex;
    std::queue<std::filesystem::path> _pgnQueue;

    std::mutex _coutMutex;
    std::atomic_int _latestGamesNumber;

    std::atomic_int _totalFileCount;
    std::atomic_int _totalGameCount;

    std::deque<Vocabulary> _vocabularies;
};

int main(int argc, char* argv[])
{
    std::string inputDirectory;
    std::string outputDirectory;
    int threadCount;
    bool commentary;

    try
    {
        TCLAP::CmdLine cmd("ChessCoachPgnToGames: Converts PGN databases to games to use in training and testing ChessCoach", ' ', "0.9");

        TCLAP::ValueArg<std::string> inputDirectoryArg("i", "input", "Input directory where PGN files are located", true /* req */, "", "string");
        TCLAP::ValueArg<std::string> outputDirectoryArg("o", "output", "Output directory where game files should be placed", true /* req */, "", "string");
        TCLAP::ValueArg<int> threadCountArg("t", "threads", "Number of threads to use (0 = autodetect)", false /* req */, 0, "number");
        TCLAP::SwitchArg commentaryArg("c", "commentary", "Parse commentary/variations and output comments", false, nullptr);

        // Usage/help seems to reverse this order.
        cmd.add(commentaryArg);
        cmd.add(threadCountArg);
        cmd.add(outputDirectoryArg);
        cmd.add(inputDirectoryArg);

        cmd.parse(argc, argv);

        inputDirectory = inputDirectoryArg.getValue();
        outputDirectory = outputDirectoryArg.getValue();
        threadCount = threadCountArg.getValue();
        commentary = commentaryArg.getValue();
        
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
        return 1;
    }

    ChessCoachPgnToGames pgnToGames(inputDirectory, outputDirectory, threadCount, commentary);

    pgnToGames.PrintExceptions();
    pgnToGames.InitializeLight();

    pgnToGames.ConvertAll();

    pgnToGames.FinalizeLight();

    return 0;
}

ChessCoachPgnToGames::ChessCoachPgnToGames(const std::filesystem::path& inputDirectory,
    const std::filesystem::path& outputDirectory, int threadCount, bool commentary)
    : _inputDirectory(inputDirectory)
    , _outputDirectory(outputDirectory)
    , _threadCount(threadCount)
    , _commentary(commentary)
    , _latestGamesNumber(0)
    , _totalFileCount(0)
    , _totalGameCount(0)
{
    if (_threadCount <= 0)
    {
        _threadCount = std::thread::hardware_concurrency();
    }
}

void ChessCoachPgnToGames::InitializeLight()
{
    InitializeStockfish();
    InitializeChessCoach();
}

void ChessCoachPgnToGames::FinalizeLight()
{
    FinalizeStockfish();
}

void ChessCoachPgnToGames::ConvertAll()
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;

    // Start the converter threads.
    for (int i = 0; i < _threadCount; i++)
    {
        threads.emplace_back(&ChessCoachPgnToGames::ConvertPgns, this);
    }

    // Distribute PGN paths.
    for (const auto& entry : std::filesystem::recursive_directory_iterator(_inputDirectory))
    {
        if (entry.path().extension().string() == ".pgn")
        {
            {
                std::lock_guard lock(_pgnQueueMutex);

                _pgnQueue.emplace(entry);
            }
            _totalFileCount++;
        }
    }

    // Poison the converter threads.
    for (int i = 0; i < _threadCount; i++)
    {
        std::lock_guard lock(_pgnQueueMutex);

        _pgnQueue.emplace(std::filesystem::path());
    }

    // Wait for the converter threads to finish.
    for (std::thread& thread : threads)
    {
        thread.join();
    }

    if (_commentary)
    {
        // Combine vocabulary from threads.
        Vocabulary vocabulary{};
        for (Vocabulary& workerVocabulary : _vocabularies)
        {
            vocabulary.commentCount += workerVocabulary.commentCount;
            vocabulary.vocabulary.merge(std::move(workerVocabulary.vocabulary));
        }

        // Write out the vocabulary document.
        const std::filesystem::path vocabularyPath = (_outputDirectory / Config::TrainingNetwork.Training.VocabularyFilename);
        std::ofstream vocabularyFile = std::ofstream(vocabularyPath, std::ios::out);
        for (const std::string& comment : vocabulary.vocabulary)
        {
            vocabularyFile << comment << std::endl;
        }

        std::cout << "Wrote " << vocabulary.commentCount << " move comments, "
            << vocabulary.vocabulary.size() << " unique" << std::endl;
    }

    const float secondsTaken = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
    const float filesPerSecond = (_totalFileCount / secondsTaken);
    const float gamesPerSecond = (_totalGameCount / secondsTaken);
    std::cout << "Converted " << _totalGameCount << " games in " << _totalFileCount << " files." << std::endl;
    std::cout << "(" << secondsTaken << " seconds total, " << filesPerSecond << " files per second, " << gamesPerSecond << " games per second)" << std::endl;
}

void ChessCoachPgnToGames::ConvertPgns()
{
    const Storage storage(Config::TrainingNetwork, Config::Misc);
    std::vector<SavedGame> games;
    std::vector<SavedCommentary> gameCommentary;

    Vocabulary& vocabulary = [&]() -> decltype(auto)
    {
        std::lock_guard lock(_coutMutex);

        return _vocabularies.emplace_back();
    }();

    while (true)
    {
        std::filesystem::path pgnPath;
        int pgnGamesConverted = 0;

        // Spin waiting for a PGN.
        while (true)
        {
            std::lock_guard lock(_pgnQueueMutex);

            if (!_pgnQueue.empty())
            {
                pgnPath = _pgnQueue.front();
                _pgnQueue.pop();
                break;
            }
        }

        // Check for poison.
        if (pgnPath.empty())
        {
            break;
        }

        std::ifstream pgnFile = std::ifstream(pgnPath, std::ios::in);
        Pgn::ParsePgn(pgnFile, [&](SavedGame&& game, SavedCommentary&& commentary)
            {
                games.emplace_back(std::move(game));
                gameCommentary.emplace_back(std::move(commentary));
                pgnGamesConverted++;

                if (games.size() >= Config::Misc.Storage_GamesPerChunk)
                {
                    SaveChunk(storage, games, gameCommentary, vocabulary);
                }
            });

        {
            std::lock_guard lock(_coutMutex);

            _totalGameCount += pgnGamesConverted;
            std::cout << "Converted " << pgnPath.filename() << ": " << pgnGamesConverted << " games" << std::endl;
        }
    }

    if (!games.empty())
    {
        SaveChunk(storage, games, gameCommentary, vocabulary);
    }
}

void ChessCoachPgnToGames::SaveChunk(const Storage& storage, std::vector<SavedGame>& games,
    std::vector<SavedCommentary>& gameCommentary, Vocabulary& vocabulary)
{
    const std::filesystem::path gamePath = (_outputDirectory / storage.GenerateSimpleChunkFilename(++_latestGamesNumber));
    if (_commentary)
    {
        storage.SaveCommentary(gamePath, games, gameCommentary, vocabulary);
    }
    else
    {
        storage.SaveChunk(gamePath, games);
    }
    games.clear();
    gameCommentary.clear();
}