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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

int main(int argc, char* argv[])
{
    Bitboards::init();
    Position::init();
    Bitbases::init();

    // Initialize threads. They directly reach out to UCI options, so we need to initialize that too.
    UCI::init(Options);
    Threads.set(static_cast<size_t>(Options["Threads"]));

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
    import_array();

    PyObject* module = nullptr;
    PyObject* function = nullptr;
    PyObject* pythonImage = nullptr;
    PyObject* result = nullptr;
    PyObject* pythonValue = nullptr;
    PyObject* pythonPolicy = nullptr;
    bool success = false;

    module = PyImport_ImportModule("predict");
    if (module)
    {
        function = PyObject_GetAttrString(module, "predict");
        if (function && PyCallable_Check(function))
        {
            npy_intp dims[3]{ 12, 8, 8 };

            float(*image)[8][8]{ new float[12][8][8] };

            pythonImage = PyArray_SimpleNewFromData(
                Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT, reinterpret_cast<void*>(image));
            if (pythonImage)
            {
                result = PyObject_CallFunctionObjArgs(function, pythonImage, nullptr);
                if (result)
                {
                    pythonValue = PyTuple_GetItem(result, 0);
                    assert(PyArray_Check(pythonValue));
                    
                    pythonPolicy = PyTuple_GetItem(result, 1);
                    assert(PyArray_Check(pythonPolicy));
                    
                    PyArrayObject* pythonValueArray = reinterpret_cast<PyArrayObject*>(pythonValue);
                    float value = reinterpret_cast<float*>(PyArray_DATA(pythonValueArray))[0];
                        
                    PyArrayObject* pythonPolicyArray = reinterpret_cast<PyArrayObject*>(pythonPolicy);
                    float(*policy)[8][8] = reinterpret_cast<float(*)[8][8]>(PyArray_DATA(pythonPolicyArray));

                    float test = policy[0][1][2];
                    float test2 = policy[3][4][5];
                    
                    success = true;
                }
            }

            delete[] image;
        }
    }
    
    if (!success && PyErr_Occurred())
    {
        PyErr_Print();
    }

    Py_XDECREF(pythonPolicy);
    Py_XDECREF(pythonValue);
    Py_XDECREF(result);
    Py_XDECREF(pythonImage);
    Py_XDECREF(function);
    Py_XDECREF(module);

    // Clean up python.
    Py_FinalizeEx();

    return 0;
}
