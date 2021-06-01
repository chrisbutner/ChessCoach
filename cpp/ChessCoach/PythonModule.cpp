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
    { "show_line",  PythonModule::ShowLine, METH_VARARGS, nullptr },
    { "evaluate_parameters",  PythonModule::EvaluateParameters, METH_VARARGS, nullptr },
    { "generate_commentary_image_for_fens",  PythonModule::GenerateCommentaryImageForFens, METH_VARARGS, nullptr },
    { "generate_commentary_image_for_position",  PythonModule::GenerateCommentaryImageForPosition, METH_VARARGS, nullptr },
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

PyObject* PythonModule::LoadChunk(PyObject*/* self*/, PyObject* args)
{
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

PyObject* PythonModule::LoadGame(PyObject*/* self*/, PyObject* args)
{
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

        assert(Instance().storage);
        Instance().storage->LoadGameFromChunk(Instance()._chunkContents, gameInChunk, &Instance()._game);

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

PyObject* PythonModule::LoadPosition(PyObject*/* self*/, PyObject* args)
{
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

PyObject* PythonModule::ShowLine(PyObject*/* self*/, PyObject* args)
{
    PyObject* pythonLine;

    if (!PyArg_UnpackTuple(args, "load_position", 1, 1, &pythonLine) ||
        !pythonLine ||
        !PyBytes_Check(pythonLine))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 1 arg: line");
        return nullptr;
    }

    const Py_ssize_t size = PyBytes_GET_SIZE(pythonLine);
    const char* data = PyBytes_AS_STRING(pythonLine);
    std::string line(data, size);

    {
        NonPythonContext context;

        Instance().worker->GuiShowLine(Instance().network, line);
    }

    Py_RETURN_NONE;
}

PyObject* PythonModule::EvaluateParameters(PyObject*/* self*/, PyObject* args)
{
    PyObject* pythonNames;
    PyObject* pythonValues;
    std::map<std::string, float> parameters;

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
    for (int i = 0; i < size; i++)
    {
        parameters[PyBytes_AsString(PyList_GetItem(pythonNames, i))] =
            static_cast<float>(PyFloat_AsDouble(PyList_GetItem(pythonValues, i)));
    }

    double evaluationScore;
    {
        NonPythonContext context;

        // Propagate the provided parameters through Config and UCI search code.
        Config::Update({} /* intUpdates */, parameters /* floatUpdates */, {} /* stringUpdates */, {} /* boolUpdates */);

        // Set up worker threads (threads/parallelism may be parameters).
        assert(Instance().network);
        WorkerGroup workerGroup;
        workerGroup.Initialize(Instance().network, nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
            Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopStrengthTest);

        // Run the test.
        const std::filesystem::path epdPath = (Platform::InstallationDataPath() / "StrengthTests" / Config::Misc.Optimization_Epd);
        auto [score, total, positions, totalNodesRequired] = workerGroup.controllerWorker->StrengthTestEpd(
            workerGroup.workCoordinator.get(), epdPath, Config::Misc.Optimization_EpdMovetimeMilliseconds, Config::Misc.Optimization_EpdNodes,
            Config::Misc.Optimization_EpdFailureNodes, Config::Misc.Optimization_EpdPositionLimit, nullptr /* progress */);

        evaluationScore = totalNodesRequired;
        workerGroup.ShutDown();
    }

    // Return a scalar.
    return PyFloat_FromDouble(evaluationScore);
}

PyObject* PythonModule::GenerateCommentaryImageForFens(PyObject*/* self*/, PyObject* args)
{
    PyObject* pythonFenBefore;
    PyObject* pythonFenAfter;

    if (!PyArg_UnpackTuple(args, "generate_commentary_image_for_fens", 2, 2, &pythonFenBefore, &pythonFenAfter) ||
        !pythonFenBefore ||
        !pythonFenAfter ||
        !PyBytes_Check(pythonFenBefore) ||
        !PyBytes_Check(pythonFenAfter))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 2 args: fenBefore, fenAfter");
        return nullptr;
    }

    const Py_ssize_t sizeBefore = PyBytes_GET_SIZE(pythonFenBefore);
    const char* dataBefore = PyBytes_AS_STRING(pythonFenBefore);
    std::string fenBefore(dataBefore, sizeBefore);

    const Py_ssize_t sizeAfter = PyBytes_GET_SIZE(pythonFenAfter);
    const char* dataAfter = PyBytes_AS_STRING(pythonFenAfter);
    std::string fenAfter(dataAfter, sizeAfter);

    npy_intp imageDims[1]{ INetwork::CommentaryInputPlaneCount };
    PyObject* pythonImage = PyArray_SimpleNew(Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64);
    PyArrayObject* pythonImageArray = reinterpret_cast<PyArrayObject*>(pythonImage);
    INetwork::PackedPlane* pythonImageArrayPtr = reinterpret_cast<INetwork::PackedPlane*>(PyArray_DATA(pythonImageArray));

    {
        NonPythonContext context;

        Game game(fenBefore, {});
        game.ApplyMoveInfer(fenAfter);
        game.GenerateCommentaryImage(pythonImageArrayPtr);
    }

    return pythonImage;
}

// Relies on "LoadChunk" and "LoadGame" having been called previously.
PyObject* PythonModule::GenerateCommentaryImageForPosition(PyObject*/* self*/, PyObject* args)
{
    PyObject* pythonPosition;

    if (!PyArg_UnpackTuple(args, "generate_commentary_image_for_position", 1, 1, &pythonPosition) ||
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
    if ((position < 0) || (position >= savedGame.moveCount))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid position");
        return nullptr;
    }

    npy_intp imageDims[1]{ INetwork::CommentaryInputPlaneCount };
    PyObject* pythonImage = PyArray_SimpleNew(Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64);
    PyArrayObject* pythonImageArray = reinterpret_cast<PyArrayObject*>(pythonImage);
    INetwork::PackedPlane* pythonImageArrayPtr = reinterpret_cast<INetwork::PackedPlane*>(PyArray_DATA(pythonImageArray));

    {
        NonPythonContext context;

        // Reach the requested position.
        Game game;
        for (int i = 0; i < position; i++)
        {
            game.ApplyMove(Move(savedGame.moves[i]));
        }

        game.GenerateCommentaryImage(pythonImageArrayPtr);
    }

    return pythonImage;
}