#include <iostream>
#include <cstdlib>
#include <array>
#include <thread>

#include <Stockfish/bitboard.h>
#include <Stockfish/position.h>
#include <Stockfish/thread.h>
#include <Stockfish/tt.h>
#include <Stockfish/uci.h>
#include <Stockfish/movegen.h>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "SelfPlay.h"
#include "Threading.h"
#include "PythonNetwork.h"
#include "Config.h"

int InitializePython();
void FinalizePython();
void InitializeStockfish();
void InitializeChessCoach();
void FinalizeStockfish();
void TrainChessCoach();
void PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork);
void TrainNetwork(const SelfPlayWorker& worker, INetwork* network, int stepCount, int checkpoint);
void DebugGame();

int main(int argc, char* argv[])
{
    // TODO: May need to set TF_CPP_MIN_LOG_LEVEL to "3" to avoid TF spam during UCI use

    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();

#if DEBUG_MCTS
    DebugGame();
#else
    TrainChessCoach();
#endif

    FinalizePython();
    FinalizeStockfish();

    return 0;
}

int InitializePython()
{
    // Work around a Python crash.
    char* pythonHome;
    errno_t err = _dupenv_s(&pythonHome, nullptr, "PYTHONHOME");
    if (err || !pythonHome)
    {
        char* condaPrefix;
        errno_t err = _dupenv_s(&condaPrefix, nullptr, "CONDA_PREFIX");
        if (!err && condaPrefix)
        {
            _putenv_s("PYTHONHOME", condaPrefix);
        }
    }

    Py_Initialize();
    return 0;
}

void FinalizePython()
{
    PyGILState_Ensure();

    Py_FinalizeEx();
}

namespace PSQT
{
    void init();
}

void InitializeStockfish()
{
    // Positions need threads, and threads need UCI options, unfortunately.
    UCI::init(Options);

    PSQT::init();
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Endgames::init();

    // Positions need threads, unfortunately.
    Threads.set(static_cast<size_t>(Options["Threads"]));

    // This usually gets initialized during a search before static evaluations, but we're not searching.
    Threads.main()->contempt = SCORE_ZERO;
}

void InitializeChessCoach()
{
    Game::Initialize();

    LargePageAllocator::Initialize();
    PredictionCache::Instance.Allocate(8); // 8 GiB
}

void FinalizeStockfish()
{
    Threads.set(0);
}

void TrainChessCoach()
{
    BatchedPythonNetwork network;
    Storage storage;

    storage.LoadExistingGames();

    // Set up configuration for full training.
    const int networkCount = (Config::TrainingSteps / Config::CheckpointInterval);
    const int gamesPerNetwork = (Config::SelfPlayGames / networkCount);
    const int maxNodesPerThread = (2 * Config::MaxMoves * INetwork::MaxBranchMoves * Config::PredictionBatchSize);
    
    std::vector<SelfPlayWorker> selfPlayWorkers(Config::SelfPlayWorkerCount);
    std::vector<std::thread> selfPlayThreads;

    WorkCoordinator workCoordinator(static_cast<int>(selfPlayWorkers.size()));
    for (int i = 0; i < selfPlayWorkers.size(); i++)
    {
        std::cout << "Starting self-play thread " << (i + 1) << " of " << selfPlayWorkers.size() <<
            " (" << Config::PredictionBatchSize << " games per thread)" << std::endl;

        selfPlayThreads.emplace_back(&SelfPlayWorker::PlayGames, &selfPlayWorkers[i], std::ref(workCoordinator), &storage, &network, maxNodesPerThread);
    }

    // If existing games found, resume progress.
    int existingNetworks = storage.CountNetworks();
    int bonusGames = std::max(0, storage.GamesPlayed() - gamesPerNetwork * existingNetworks);

    for (int n = existingNetworks; n < networkCount; n++)
    {
        // Play self-play games (skip if we already have enough to train this checkpoint).
        const int actualGames = (gamesPerNetwork - bonusGames);
        if (actualGames > 0)
        {
            PlayGames(workCoordinator, actualGames);
        }
        bonusGames = std::max(0, bonusGames - gamesPerNetwork);

        // Train the network.
        const int checkpoint = (n + 1) * Config::CheckpointInterval;
        TrainNetwork(selfPlayWorkers.front(), &network, Config::CheckpointInterval, checkpoint);

        // Clear the prediction cache to prepare for the new network.
        PredictionCache::Instance.PrintDebugInfo();
        PredictionCache::Instance.Clear();
        PredictionCache::Instance.PrintDebugInfo();
    }
}

void PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork)
{
    std::cout << "Playing " << gamesPerNetwork << " games..." << std::endl;
    assert(gamesPerNetwork > 0);

    workCoordinator.ResetWorkItemsRemaining(gamesPerNetwork);

    workCoordinator.WaitForWorkers();
}

void TrainNetwork(const SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint)
{
    std::cout << "Training..." << std::endl;

    selfPlayWorker.TrainNetwork(network, stepCount, checkpoint);
}

void DebugGame()
{
    Storage storage;

    SelfPlayWorker worker;
    worker.Initialize(&storage, 2 * Config::MaxMoves * INetwork::MaxBranchMoves * Config::PredictionBatchSize);

    BatchedPythonNetwork network;

    StoredGame stored = storage.LoadFromDisk("path_to_game");
    worker.DebugGame(&network, 0, stored, 23);
}