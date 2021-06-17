#include "PythonModule.h"

#include <sstream>
#include <iomanip>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <Stockfish/uci.h>

#include "Pgn.h"

PyMethodDef PythonModule::ChessCoachMethods[] = {
    { "load_chunk",  PythonModule::LoadChunk, METH_VARARGS, nullptr },
    { "load_game",  PythonModule::LoadGame, METH_VARARGS, nullptr },
    { "load_position",  PythonModule::LoadPosition, METH_VARARGS, nullptr },
    { "show_line",  PythonModule::ShowLine, METH_VARARGS, nullptr },
    { "evaluate_parameters",  PythonModule::EvaluateParameters, METH_VARARGS, nullptr },
    { "generate_commentary_image_for_fens",  PythonModule::GenerateCommentaryImageForFens, METH_VARARGS, nullptr },
    { "generate_commentary_image_for_position",  PythonModule::GenerateCommentaryImageForPosition, METH_VARARGS, nullptr },
    { "bot_search",  PythonModule::BotSearch, METH_VARARGS, nullptr },
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

        assert(Instance().workerGroup);
        Instance().workerGroup->controllerWorker->GuiShowLine(Instance().network, line);
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

        // Set up worker threads (threads/parallelism may be parameters, so we can't use a long-lived WorkerGroup).
        assert(Instance().network);
        assert(!Instance().workerGroup);
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

PyObject* PythonModule::BotSearch(PyObject*/* self*/, PyObject* args)
{
    PyObject* pythonGameId;
    PyObject* pythonFen;
    PyObject* pythonMoves;
    PyObject* pythonBotSide;
    PyObject* pythonLimitSeconds;
    PyObject* pythonWtime;
    PyObject* pythonBtime;
    PyObject* pythonWinc;
    PyObject* pythonBinc;

    if (!PyArg_UnpackTuple(args, "bot_search", 9, 9, &pythonGameId, &pythonFen, &pythonMoves, &pythonBotSide, &pythonLimitSeconds,
        &pythonWtime, &pythonBtime, &pythonWinc, &pythonBinc) ||
        !pythonFen || !pythonMoves || !pythonBotSide || !pythonLimitSeconds || !pythonWtime || !pythonBtime || !pythonWinc || !pythonBinc ||
        !PyBytes_Check(pythonGameId) || !PyBytes_Check(pythonFen) || !PyBytes_Check(pythonMoves) || !PyLong_Check(pythonBotSide) || !PyLong_Check(pythonLimitSeconds) ||
        !PyLong_Check(pythonWtime) || !PyLong_Check(pythonBtime) || !PyLong_Check(pythonWinc) || !PyLong_Check(pythonBinc))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 9 args: game_id, fen, moves, bot_side, limit_seconds, wtime, btime, winc, binc");
        return nullptr;
    }

    const Py_ssize_t sizeGameId = PyBytes_GET_SIZE(pythonGameId);
    const char* dataGameId = PyBytes_AS_STRING(pythonGameId);
    std::string gameId(dataGameId, sizeGameId);

    const Py_ssize_t sizeFen = PyBytes_GET_SIZE(pythonFen);
    const char* dataFen = PyBytes_AS_STRING(pythonFen);
    std::string fen(dataFen, sizeFen);

    const Py_ssize_t sizeMoves = PyBytes_GET_SIZE(pythonMoves);
    const char* dataMoves = PyBytes_AS_STRING(pythonMoves);
    std::string moves(dataMoves, sizeMoves);

    const Color botSide = Color(PyLong_AsLong(pythonBotSide));
    const int limitSeconds = PyLong_AsLong(pythonLimitSeconds);
    const int wtime = PyLong_AsLong(pythonWtime);
    const int btime = PyLong_AsLong(pythonBtime);
    const int winc = PyLong_AsLong(pythonWinc);
    const int binc = PyLong_AsLong(pythonBinc);

    std::string status;
    int ply;
    std::string san;
    std::vector<std::string> comments;

    {
        NonPythonContext context;

        assert(Instance().network);
        assert(Instance().workerGroup);

        // Stop and ready workers.
        Instance().workerGroup->workCoordinator->ResetWorkItemsRemaining(0);
        Instance().workerGroup->workCoordinator->WaitForWorkers();

        // If there's no game ID then we just needed to stop any search/ponder in progress.
        if (gameId.empty())
        {
            Py_RETURN_NONE;
        }

        // Set up the position and parse moves. Also generate a commentary image while we have the game set up,
        // as long as we have enough think time remaining.
        const std::string& actualFen = ((fen == "startpos") ? Game::StartingPosition : fen);
        std::vector<Move> parsedMoves;
        Game game(actualFen, {});
        std::stringstream uciMoves(moves);
        std::string token;
        while (uciMoves >> token)
        {
            const Move move = UCI::to_move(game.GetPosition(), token);
            if (move == MOVE_NONE)
            {
                break;
            }
            game.ApplyMove(move);
        }
        std::unique_ptr<INetwork::CommentaryInputPlanes> commentaryImage;
        const int botRemainingMilliseconds = ((botSide == WHITE) ? wtime : btime);
        if (botRemainingMilliseconds >= Config::Misc.Bot_CommentaryMinimumRemainingMilliseconds)
        {
            commentaryImage.reset(new INetwork::CommentaryInputPlanes());
            game.GenerateCommentaryImage(commentaryImage->data()); // Relies on "game.Moves()", so run before the "std::move"
        }
        parsedMoves = std::move(game.Moves());

        // Propagate the position.
        Instance().workerGroup->controllerWorker->SearchUpdatePosition(actualFen, parsedMoves, false /* forceNewPosition */);

        // Check whether to search or ponder and set up time control.
        const bool search = (game.ToPlay() == botSide);
        TimeControl timeControl = {};
        if (search)
        {
            timeControl.timeRemainingMs[WHITE] = wtime;
            timeControl.timeRemainingMs[BLACK] = btime;
            timeControl.incrementMs[WHITE] = winc;
            timeControl.incrementMs[BLACK] = binc;

            // It's best to override limits via wtime/btime still so that elimination and safety buffer continue to work.
            if (limitSeconds)
            {
                const int64_t limitRemainingMs = (Config::Misc.TimeControl_FractionOfRemaining * static_cast<int64_t>(limitSeconds) * 1000);
                timeControl.timeRemainingMs[WHITE] = std::min(timeControl.timeRemainingMs[WHITE], limitRemainingMs);
                timeControl.timeRemainingMs[BLACK] = std::min(timeControl.timeRemainingMs[BLACK], limitRemainingMs);
            }

            status = "searching";
        }
        else
        {
            // Opponent's move, so "ponder".
            timeControl.infinite = true;
            status = "pondering";
        }
        Instance().workerGroup->searchState.Reset(timeControl);
        Instance().workerGroup->searchState.botGameId = gameId;

        // Comment on the position and return the last SAN, as long as there's been at least one move,
        // since the encoder is trained with before-and-after positions (and as long as there's enough
        // think time remaining).
        //
        // It's important to do this before waking up search workers so that this call can
        // return to Python quickly. Assume that the additional bot safety buffer covers the time needed.
        //
        // This acquires a "PythonContext" in the PythonNetwork call, so make it here in the
        // "NonPythonContext" so we that don't over-release (even though it's GIL-inefficient).
        ply = game.Ply();
        if (!parsedMoves.empty() && commentaryImage)
        {
            Position& position = game.GetPosition();
            const Move lastMove = parsedMoves.back();
            position.undo_move(lastMove);
            san = Pgn::San(position, lastMove, true /* showCheckmate */);
            comments = Instance().network->PredictCommentaryBatch(1, commentaryImage.get());
        }

        // Start searching/pondering, but don't wait for workers.
        // Return to the bot loop in Python and let the primary search worker
        // send the best move asynchronously (like UCI "bestmove" to stdout).
        Instance().workerGroup->workCoordinator->ResetWorkItemsRemaining(1);
    }

    // Pack and return a 4-tuple.
    PyObject* pythonStatus = PyUnicode_FromStringAndSize(status.data(), status.length());
    PyObject* pythonPly = PyLong_FromLong(ply);
    PyObject* pythonSan = (!san.empty()
        ? PyUnicode_FromStringAndSize(san.data(), san.length())
        : (Py_INCREF(Py_None), Py_None));
    PyObject* pythonComment = (!comments.empty()
        ? PyUnicode_FromStringAndSize(comments[0].data(), comments[0].length())
        : (Py_INCREF(Py_None), Py_None));
    PyObject* pythonTuple = PyTuple_Pack(4, pythonStatus, pythonPly, pythonSan, pythonComment);
    PythonNetwork::PyAssert(pythonTuple);
    Py_DECREF(pythonStatus);
    Py_DECREF(pythonPly);
    Py_DECREF(pythonSan);
    Py_DECREF(pythonComment);
    return pythonTuple;
}