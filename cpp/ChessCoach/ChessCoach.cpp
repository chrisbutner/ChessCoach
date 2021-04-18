#include "ChessCoach.h"

#include <cstdlib>
#include <exception>
#include <iostream>

#include <Stockfish/bitboard.h>
#include <Stockfish/position.h>
#include <Stockfish/thread.h>
#include <Stockfish/uci.h>

#include "PythonNetwork.h"
#include "PythonModule.h"
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
            }
            std::_Exit(1);
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

    if (PyImport_AppendInittab(PythonModule::ChessCoachModule.m_name, PythonModule::PyInit_ChessCoachModule) == -1)
    {
        throw std::runtime_error("Failed to register 'chesscoach' module");
    }

    Py_Initialize();

    // Initialize numpy.
    if (PY_ARRAY_UNIQUE_SYMBOL == nullptr)
    {
        _import_array();
    }

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
    PredictionCache::Instance.Allocate(Config::Misc.PredictionCache_RequestGibibytes, Config::Misc.PredictionCache_MinGibibytes);
}

// Keep Python visibility isolated to the ChessCoach library.
void ChessCoach::InitializePythonModule(Storage* storage, WorkerGroup* workerGroup)
{
    PythonModule::Instance().storage = storage;
    PythonModule::Instance().workerGroup = workerGroup;
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
    return new PythonNetwork();
}

// Keep Python visibility isolated to the ChessCoach library.
void ChessCoach::OptimizeParameters()
{
    // See "ChessCoachOptimizeParameters::Run" for rationale.
    // This runs forever, so no need to clean anything up.
    PythonContext context;

    PyObject* sys = PyImport_ImportModule("sys");
    PythonNetwork::PyAssert(sys);
    PyObject* sysPath = PyObject_GetAttrString(sys, "path");
    PythonNetwork::PyAssert(sysPath);
    PyObject* pythonPath = PyUnicode_FromString(Platform::InstallationScriptPath().string().c_str());
    PythonNetwork::PyAssert(pythonPath);
    const int error = PyList_Append(sysPath, pythonPath);
    PythonNetwork::PyAssert(!error);

    PyObject* module = PyImport_ImportModule("optimization");
    PythonNetwork::PyAssert(module);

    PyObject* optimizeParametersFunction = PyObject_GetAttrString(module, "optimize_parameters");
    PythonNetwork::PyAssert(optimizeParametersFunction);
    PythonNetwork::PyAssert(PyCallable_Check(optimizeParametersFunction));

    PyObject* result = PyObject_CallFunctionObjArgs(optimizeParametersFunction, nullptr);
    PythonNetwork::PyAssert(result);
}