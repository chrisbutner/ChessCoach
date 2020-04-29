#include "ChessCoach.h"

#include <cstdlib>
#include <exception>
#include <iostream>

#include <Stockfish/bitboard.h>
#include <Stockfish/position.h>
#include <Stockfish/thread.h>
#include <Stockfish/uci.h>

#include "PythonNetwork.h"
#include "Game.h"
#include "PredictionCache.h"
#include "PoolAllocator.h"

namespace PSQT
{
    void init();
}

void ChessCoach::PrintExceptions()
{
    std::set_terminate([]
        {
            auto exception = std::current_exception();
            if (exception)
            {
                try
                {
                    std::rethrow_exception(exception);
                }
                catch (const std::exception& e)
                {
                    std::cout << "Unhandled exception: " << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cout << "Unhandled exception (unknown type)" << std::endl;
                }
                std::abort();
            }
            else
            {
                // Finished running, graceful termination.
            }
        });
}

void ChessCoach::Initialize()
{
    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();
    InitializePredictionCache();
}

void ChessCoach::Finalize()
{
    FinalizePython();
    FinalizeStockfish();
}

int ChessCoach::InitializePython()
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

    // Release the implicit main-thread GIL.
    PythonContext context;

    return 0;
}

void ChessCoach::InitializeStockfish()
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

void ChessCoach::InitializeChessCoach()
{
    Config::Initialize();
    Game::Initialize();
}

void ChessCoach::InitializePredictionCache()
{
    PredictionCache::Instance.Allocate(Config::Misc.PredictionCache_SizeGb);
}
void ChessCoach::FinalizePython()
{
    PyGILState_Ensure();

    Py_FinalizeEx();
}

void ChessCoach::FinalizeStockfish()
{
    Threads.set(0);
}

INetwork* ChessCoach::CreateNetwork(const NetworkConfig& networkConfig) const
{
    INetwork* network = new PythonNetwork();
    network->LoadNetwork(networkConfig.Name.c_str());
    return network;
}