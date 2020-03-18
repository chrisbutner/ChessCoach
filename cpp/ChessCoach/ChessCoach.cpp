#include <iostream>
#include <cstdlib>

#include "bitboard.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "movegen.h"

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

int main(int argc, char* argv[])
{
    Bitboards::init();
    Position::init();
    Bitbases::init();

    // Initialize threads. They directly reach out to UCI options, so we need to initialize that too.
    UCI::init(Options);
    Threads.set(Options["Threads"]);

    // Set up the starting position.
    const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    StateListPtr states(new std::deque<StateInfo>(1));
    Position position;
    position.set(StartFEN, false /* isChess960 */, &states->back(), Threads.main());  

    // Generate legal moves.
    ExtMove moves[MAX_MOVES];
    ExtMove* cur = moves;
    ExtMove* endMoves = generate<LEGAL>(position, cur);

    // Debug
    std::cout << "Legal moves: " << (endMoves - cur) << std::endl;
    while (cur != endMoves)
    {
        std::cout << from_sq(cur->move) << " to " << to_sq(cur->move) << std::endl;
        cur++;
    }

    // Clean up.
    Threads.set(0);

    // Test Python

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

    PyObject* module = PyImport_ImportModule("predict");
    if (module != nullptr)
    {
        PyObject* function = PyObject_GetAttrString(module, "predict");
        if (function && PyCallable_Check(function))
        {
            // TODO: Args
            PyObject* result = PyObject_CallObject(function, nullptr);
            if (result)
            {
                // TODO: Use result
                Py_DECREF(result);
            }
            else
            {
                PyErr_Print();
            }
        }
        else
        {
            if (PyErr_Occurred())
            {
                PyErr_Print();
            }
        }
        Py_XDECREF(function);
        Py_DECREF(module);
    }
    else
    {
        PyErr_Print();
    }
    Py_FinalizeEx();

    return 0;
}
