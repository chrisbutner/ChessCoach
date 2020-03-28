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
void InitializeChess();
void FinalizeStockfish();
void TrainChessCoach();
void PlayGames(const std::vector<Mcts>& workers, INetwork* network, int gamesPerNetwork);
void TrainNetwork(const Mcts& worker, INetwork* network, int stepCount, int checkpoint);

int main(int argc, char* argv[])
{
    // TODO: May need to set TF_CPP_MIN_LOG_LEVEL to "3" to avoid TF spam during UCI use

    InitializePython();
    InitializeStockfish();
    InitializeChess();

    TrainChessCoach();

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
    Py_FinalizeEx();
}

void InitializeStockfish()
{
    Bitboards::init();
    Position::init();
    Bitbases::init();

    // Initialize threads. They directly reach out to UCI options, so we need to initialize that too.
    UCI::init(Options);
    Threads.set(static_cast<size_t>(Options["Threads"]));
}

void InitializeChess()
{
    Game::Initialize();
}

void FinalizeStockfish()
{
    Threads.set(0);
}

void TrainChessCoach()
{
    std::unique_ptr<BatchedPythonNetwork> network(new BatchedPythonNetwork());
    std::thread networkBatcher(&BatchedPythonNetwork::Work, network.get()); // TODO: Lasts too long

    Storage storage;
    storage.LoadExistingGames();

    std::vector<Mcts> selfPlayWorkers;
    const int selfPlayWorkerCount =
#ifdef _DEBUG
        1;
#else
        32;
#endif
    for (int i = 0; i < selfPlayWorkerCount; i++)
    {
        selfPlayWorkers.emplace_back(&storage);
    }

    // Set up configuration for full training.
    const int networkCount = (Config::TrainingSteps / Config::CheckpointInterval);
    const int gamesPerNetwork = (Config::SelfPlayGames / networkCount);

    // If existing games found, resume progress.
    int existingNetworks = storage.CountNetworks();
    int bonusGames = std::max(0, storage.GamesPlayed() - gamesPerNetwork * existingNetworks);

    for (int n = existingNetworks; n < networkCount; n++)
    {
        PlayGames(selfPlayWorkers, network.get(), gamesPerNetwork - bonusGames);
        bonusGames = std::max(0, bonusGames - gamesPerNetwork);

        const int checkpoint = (n + 1) * Config::CheckpointInterval;
        TrainNetwork(selfPlayWorkers.front(), network.get(), Config::CheckpointInterval, checkpoint);
    }
}

void PlayGames(const std::vector<Mcts>& selfPlayWorkers, INetwork* network, int gamesPerNetwork)
{
    std::vector<std::thread> selfPlayThreads;

    WorkCoordinator workCoordinator(gamesPerNetwork);
    network->SetEnabled(true);
    for (int i = 0; i < selfPlayWorkers.size(); i++)
    {
        std::cout << "Starting self-play thread " << (i + 1) << std::endl;

        selfPlayThreads.emplace_back(&Mcts::PlayGames, &selfPlayWorkers[i], std::ref(workCoordinator), network);
    }

    workCoordinator.Wait();
    network->SetEnabled(false);

    for (int i = 0; i < selfPlayWorkers.size(); i++)
    {
        selfPlayThreads[i].join();
    }
}

void TrainNetwork(const Mcts& worker, INetwork* network, int stepCount, int checkpoint)
{
    worker.TrainNetwork(network, stepCount, checkpoint);
}