#include "PythonModule.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <sstream>
#include <iomanip>

#include "Pgn.h"

PyMethodDef PythonModule::ChessCoachMethods[] = {
    { "load_chunk",  PythonModule::LoadChunk, METH_VARARGS, nullptr },
    { "load_game",  PythonModule::LoadGame, METH_VARARGS, nullptr },
    { "load_position",  PythonModule::LoadPosition, METH_VARARGS, nullptr },
    { "evaluate_parameters",  PythonModule::EvaluateParameters, METH_VARARGS, nullptr },
    { nullptr, nullptr, 0, nullptr }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
PyModuleDef PythonModule::ChessCoachModule = {
    PyModuleDef_HEAD_INIT,
    "chesscoach",
    nullptr,
    -1,
    ChessCoachMethods
};
#pragma GCC diagnostic pop

PyObject* PythonModule::PyInit_ChessCoachModule()
{
    return PyModule_Create(&ChessCoachModule);
}

PythonModule& PythonModule::Instance()
{
    static PythonModule instance;
    return instance;
}

PyObject* PythonModule::LoadChunk(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* pythonBytes;

    if (!PyArg_UnpackTuple(args, "load_chunk", 1, 1, &pythonBytes) ||
        !pythonBytes ||
        !PyBytes_Check(pythonBytes))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 1 args: bytes");
        return nullptr;
    }

    const Py_ssize_t size = PyBytes_GET_SIZE(pythonBytes);
    const char* data = PyBytes_AS_STRING(pythonBytes);

    Instance()._chunkContents = std::string(data, size);

    Py_RETURN_NONE;
}

PyObject* PythonModule::LoadGame(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* pythonGameInChunk;

    if (!PyArg_UnpackTuple(args, "load_game", 1, 1, &pythonGameInChunk) ||
        !pythonGameInChunk ||
        !PyLong_Check(pythonGameInChunk))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 1 arg: game_in_chunk");
        return nullptr;
    }

    const int gameInChunk = PyLong_AsLong(pythonGameInChunk);

    std::stringstream pgn;
    {
        NonPythonContext context;

        assert(Instance().Storage);
        Instance().Storage->LoadGameFromChunk(Instance()._chunkContents, gameInChunk, &Instance()._game);

        Pgn::GeneratePgn(pgn, Instance()._game);
    }

    PyObject* pythonPositionCount = PyLong_FromLong(Instance()._game.moveCount);
    PyObject* pythonPgn = PyUnicode_FromStringAndSize(pgn.str().data(), pgn.str().length());


    // Pack and return a 2-tuple.
    PyObject* pythonTuple = PyTuple_Pack(2, pythonPositionCount, pythonPgn);
    PythonNetwork::PyAssert(pythonTuple);
    Py_DECREF(pythonPositionCount);
    Py_DECREF(pythonPgn);
    return pythonTuple;
}

PyObject* PythonModule::LoadPosition(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* pythonPosition;

    if (!PyArg_UnpackTuple(args, "load_position", 1, 1, &pythonPosition) ||
        !pythonPosition ||
        !PyLong_Check(pythonPosition))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 1 arg: position");
        return nullptr;
    }

    // Allow Python-style "-1" for the final position, etc.
    const SavedGame& savedGame = Instance()._game;
    int position = PyLong_AsLong(pythonPosition);
    if (position < 0)
    {
        position += savedGame.moveCount;
    }
    position = std::clamp(position, 0, savedGame.moveCount - 1);

    std::string fen;
    std::stringstream evaluation;
    std::vector<std::string> sans;
    std::vector<std::string> froms;
    std::vector<std::string> tos;
    std::vector<float> policyValues;
    {
        NonPythonContext context;

        // Reach the requested position.
        Game game;
        for (int i = 0; i < position; i++)
        {
            game.ApplyMove(Move(savedGame.moves[i]));
        }

        fen = game.GetPosition().fen();
        evaluation << std::fixed << std::setprecision(6) << savedGame.mctsValues[position]
            << " (" << (Game::ProbabilityToCentipawns(savedGame.mctsValues[position]) / 100.f) << " pawns)";
        for (const auto& [move, value] : savedGame.childVisits[position])
        {
            sans.emplace_back(Pgn::San(game.GetPosition(), move, true /* showCheckmate */));
            froms.emplace_back(Game::SquareName[from_sq(move)]);
            tos.emplace_back(Game::SquareName[to_sq(move)]);
            policyValues.push_back(value);
        }
    }

    PyObject* pythonClampedPosition = PyLong_FromLong(position);
    PyObject* pythonFen = PyUnicode_FromStringAndSize(fen.data(), fen.size());
    PyObject* pythonEvaluation = PyUnicode_FromString(evaluation.str().c_str());
    PyObject* pythonSans = PythonNetwork::PackNumpyStringArray(sans);
    PyObject* pythonFroms = PythonNetwork::PackNumpyStringArray(froms);
    PyObject* pythonTos = PythonNetwork::PackNumpyStringArray(tos);

    npy_intp policyValueDims[1]{ static_cast<int>(policyValues.size()) };
    PyObject* pythonPolicyValues = PyArray_SimpleNew(
        Py_ARRAY_LENGTH(policyValueDims), policyValueDims, NPY_FLOAT32);
    PythonNetwork::PyAssert(pythonPolicyValues);
    float* pythonPolicyValuesData = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pythonPolicyValues)));
    std::copy(policyValues.begin(), policyValues.end(), pythonPolicyValuesData);

    // Pack and return a 7-tuple.
    PyObject* pythonTuple = PyTuple_Pack(7, pythonClampedPosition, pythonFen, pythonEvaluation, pythonSans, pythonFroms, pythonTos, pythonPolicyValues);
    PythonNetwork::PyAssert(pythonTuple);
    Py_DECREF(pythonClampedPosition);
    Py_DECREF(pythonFen);
    Py_DECREF(pythonEvaluation);
    Py_DECREF(pythonSans);
    Py_DECREF(pythonFroms);
    Py_DECREF(pythonTos);
    Py_DECREF(pythonPolicyValues);
    return pythonTuple;
}

PyObject* PythonModule::EvaluateParameters(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* pythonNames;
    PyObject* pythonValues;
    std::vector<std::string> names;
    std::vector<float> values;

    if (!PyArg_UnpackTuple(args, "evaluate_parameters", 2, 2, &pythonNames, &pythonValues) ||
        !pythonNames ||
        !pythonValues ||
        !PyList_Check(pythonNames) ||
        !PyList_Check(pythonValues))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 2 args: names, values");
        return nullptr;
    }

    const Py_ssize_t size = PyList_Size(pythonNames);
    names.resize(size);
    values.resize(size);
    for (int i = 0; i < size; i++)
    {
        names[i] = PyBytes_AsString(PyList_GetItem(pythonNames, i));
        values[i] = static_cast<float>(PyFloat_AsDouble(PyList_GetItem(pythonValues, i)));
    }

    double evaluationScore;
    double evaluationError;
    {
        NonPythonContext context;

        // TODO: Apply params

        // Use the faster student network, matching UCI.
        assert(Instance().Worker);
        assert(Instance().Network);
        const std::filesystem::path epdPath = (Platform::InstallationDataPath() / "StrengthTests" / Config::Misc.Optimization_Epd);
        auto [score, total, positions, totalNodesRequired] = Instance().Worker->StrengthTestEpd(
            Instance().Network, NetworkType_Student, epdPath, 0 /* moveTimeMs */, Config::Misc.Optimization_Nodes,
            Config::Misc.Optimization_FailureNodes, Config::Misc.Optimization_PositionLimit, nullptr /* progress */);

        // Wild guess at error for now: too expensive to run multiple samples.
        evaluationScore = totalNodesRequired;
        evaluationError = 0.0001 * Config::Misc.Optimization_FailureNodes * Config::Misc.Optimization_PositionLimit;
    }

    PyObject* pythonScore = PyFloat_FromDouble(evaluationScore);
    PyObject* pythonError = PyFloat_FromDouble(evaluationError);

    // Pack and return a 2-tuple.
    PyObject* pythonTuple = PyTuple_Pack(2, pythonScore, pythonError);
    PythonNetwork::PyAssert(pythonTuple);
    Py_DECREF(pythonScore);
    Py_DECREF(pythonError);
    return pythonTuple;
}