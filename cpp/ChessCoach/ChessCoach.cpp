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
    // Handle repeated initialization during tests.
    if (Py_IsInitialized())
    {
        // Otherwise, we don't hold the GIL, and PySys_SetArgvEx->PySys_SetObject blows up on null thread state.
        return;
    }

    // Python path/home detection is broken on Windows. Work around it by relying on the CHESSCOACH_PYTHONHOME
    // variable set in "setup.cmd", which will run "activate_virtual_env.cmd" before "where python", and copying this
    // to PYTHONHOME before initializing Python. There's also "Py_SetPythonHome" but this is simpler than decoding
    // to a wide string and keeping static storage around.
    const std::string pythonHome = Platform::GetEnvironmentVariable("PYTHONHOME");
    if (pythonHome.empty())
    {
        const std::string chessCoachPythonHome = Platform::GetEnvironmentVariable("CHESSCOACH_PYTHONHOME");
        if (!chessCoachPythonHome.empty())
        {
            Platform::SetEnvironmentVariable("PYTHONHOME", chessCoachPythonHome.c_str());
        }
    }

    if (PyImport_AppendInittab(PythonModule::ChessCoachModule.m_name, PythonModule::PyInit_ChessCoachModule) == -1)
    {
        throw ChessCoachException("Failed to register 'chesscoach' module");
    }

    // Pass initsigs=0 so that Python doesn't take over signal handling.
    Py_InitializeEx(0 /* initsigs */);

    // Initialize argv[0] so that TensorFlow and Matplotlib don't crash, but don't pass in any actual arguments.
    const wchar_t* empty[] = { L"" };
    PySys_SetArgvEx(0, const_cast<wchar_t**>(empty), 0 /* updatepath */);

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
    PredictionCache::Instance.Allocate(Config::Misc.PredictionCache_SizeMebibytes);
}

// Keep Python visibility isolated to the ChessCoach library.
void ChessCoach::InitializePythonModule(Storage* storage, INetwork* network, WorkerGroup* workerGroup)
{
    PythonModule::Instance().storage = storage;
    PythonModule::Instance().network = network;
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