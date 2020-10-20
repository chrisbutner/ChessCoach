#include "ChessCoach.h"

#include <cstdlib>
#include <exception>
#include <iostream>

#include <Stockfish/bitboard.h>
#include <Stockfish/position.h>
#include <Stockfish/thread.h>
#include <Stockfish/uci.h>

#include "PythonNetwork.h"
#undef NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Game.h"
#include "PredictionCache.h"
#include "PoolAllocator.h"
#include "Platform.h"

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

void ChessCoach::InitializePython()
{
    // Work around a Python crash.
    const std::string pythonHome = Platform::GetEnvironmentVariable("PYTHONHOME");
    if (pythonHome.empty())
    {
        const std::string condaPrefix = Platform::GetEnvironmentVariable("CONDA_PREFIX");
        if (!condaPrefix.empty())
        {
            Platform::SetEnvironmentVariable("PYTHONHOME", condaPrefix.c_str());
        }
    }

    Py_Initialize();

    // Initialize numpy.
    _import_array();

    // Release the implicit main-thread GIL.
    PythonContext context;
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
    int ignoreStepCount;
    return CreateNetworkWithInfo(networkConfig, ignoreStepCount);
}

INetwork* ChessCoach::CreateNetworkWithInfo(const NetworkConfig& networkConfig, int& stepCountOut) const
{
    INetwork* network = new PythonNetwork();
    stepCountOut = network->LoadNetwork(networkConfig.Name.c_str());
    return network;
}