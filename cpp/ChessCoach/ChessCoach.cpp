#include "ChessCoach.h"

#include <cstdlib>

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

void ChessCoach::Initialize()
{
    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();
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
    Game::Initialize();

    LargePageAllocator::Initialize();
    PredictionCache::Instance.Allocate(8 /* sizeGb */);
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

INetwork* ChessCoach::CreateNetwork() const
{
    return new BatchedPythonNetwork();
}