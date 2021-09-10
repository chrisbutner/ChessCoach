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
    PyObject* pythonCanComment;
    PyObject* pythonLimitSeconds;
    PyObject* pythonWtime;
    PyObject* pythonBtime;
    PyObject* pythonWinc;
    PyObject* pythonBinc;

    if (!PyArg_UnpackTuple(args, "bot_search", 10, 10, &pythonGameId, &pythonFen, &pythonMoves, &pythonBotSide, &pythonCanComment, &pythonLimitSeconds,
        &pythonWtime, &pythonBtime, &pythonWinc, &pythonBinc) ||
        !pythonFen || !pythonMoves || !pythonBotSide || !pythonCanComment || !pythonLimitSeconds || !pythonWtime || !pythonBtime || !pythonWinc || !pythonBinc ||
        !PyBytes_Check(pythonGameId) || !PyBytes_Check(pythonFen) || !PyBytes_Check(pythonMoves) || !PyLong_Check(pythonBotSide) || !PyBool_Check(pythonCanComment) || !PyLong_Check(pythonLimitSeconds) ||
        !PyLong_Check(pythonWtime) || !PyLong_Check(pythonBtime) || !PyLong_Check(pythonWinc) || !PyLong_Check(pythonBinc))
    {
        PyErr_SetString(PyExc_TypeError, "Expected 10 args: game_id, fen, moves, bot_side, can_comment, limit_seconds, wtime, btime, winc, binc");
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
    const bool canComment = PyObject_IsTrue(pythonCanComment);
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

        // Signal workers to stop.
        Instance().workerGroup->workCoordinator->ResetWorkItemsRemaining(0);

        // If there's no game ID then we just needed to stop any search/ponder in progress.
        if (gameId.empty())
        {
            // Finish stopping and readying workers.
            Instance().workerGroup->workCoordinator->WaitForWorkers();

            Py_RETURN_NONE;
        }

        // Set up the position and parse moves.
        const std::string& actualFen = ((fen == "startpos") ? Game::StartingPosition : fen);
        Game game(actualFen, {});
        std::stringstream uciMoves(moves);
        std::string token;
        while (uciMoves >> token)
        {
            Move move = UCI::to_move(game.GetPosition(), token);
            if (move == MOVE_NONE)
            {
                // We may get Chess960 castling notation for positions set up with a FEN for material odds.
                if (token == "e1h1") move = make<CASTLING>(SQ_E1, SQ_H1);
                else if (token == "e1a1") move = make<CASTLING>(SQ_E1, SQ_A1);
                else if (token == "e8h8") move = make<CASTLING>(SQ_E8, SQ_H8);
                else if (token == "e8a8") move = make<CASTLING>(SQ_E8, SQ_A8);
                else break;
            }
            game.ApplyMove(move);
        }
        ply = game.Ply();

        // Finish stopping and readying workers.
        Instance().workerGroup->workCoordinator->WaitForWorkers();

        // Propagate the position.
        Instance().workerGroup->controllerWorker->SearchUpdatePosition(actualFen, game.Moves(), false /* forceNewPosition */);

        // Check whether to search or ponder and set up time control.
        // Protect against dropping to increment-only and losing to a lag spike by using the overall safety buffer.
        assert(Config::Misc.TimeControl_SafetyBufferOverallMilliseconds >= 1000);
        bool skipSearch = false;
        const bool search = (game.ToPlay() == botSide);
        TimeControl timeControl = {};
        timeControl.timeRemainingMs[WHITE] = std::max(1, wtime - Config::Misc.TimeControl_SafetyBufferOverallMilliseconds); // Zero means "no limit".
        timeControl.timeRemainingMs[BLACK] = std::max(1, btime - Config::Misc.TimeControl_SafetyBufferOverallMilliseconds); // Zero means "no limit".
        timeControl.incrementMs[WHITE] = winc;
        timeControl.incrementMs[BLACK] = binc;

        // Add additional safety buffer to compensate for network calls, commentary,
        // amortizing ponder pruning, and lack of progress in no-increment games.
        static int originalBuffer = Config::Misc.TimeControl_SafetyBufferMoveMilliseconds;
        assert(Config::Misc.Bot_PonderBufferMinMilliseconds <= Config::Misc.Bot_PonderBufferMaxMilliseconds);
        const int ponderBuffer = std::clamp(
            static_cast<int>(Config::Misc.Bot_PonderBufferProportion * timeControl.timeRemainingMs[botSide]),
            Config::Misc.Bot_PonderBufferMinMilliseconds,
            Config::Misc.Bot_PonderBufferMaxMilliseconds);
        Config::Misc.TimeControl_SafetyBufferMoveMilliseconds = (originalBuffer + ponderBuffer);

        // Lichess only allows ~3 moves per second, plus some burst. Going over the limit
        // means getting hit with a 429 error and having to wait 1 minute, effectively losing.
        // Try to split the difference, allowing for some deallocation and request overhead,
        // but also eating into some burst allowance as remaining time nears zero.
        assert(Config::Misc.TimeControl_AbsoluteMinimumMilliseconds >= 150);

        if (search)
        {
            // Limit the first few plies to avoid a game abort. The first limit reached will stop the search.
            if (limitSeconds)
            {
                timeControl.moveTimeMs = (limitSeconds * 1000);
            }

            status = "searching";
        }
        else
        {
            // Opponent's move, so "ponder". Pondering adds complications for tree search because pruning the tree
            // via deallocations can take time. Normally this only consumes "free time", but when told to stop pondering
            // and start searching, in a real-time bot context, this eats into actual search time.
            //
            // This is an even bigger problem right now because of the non-pooled variably-sized Node[] allocations, which can
            // take ~60+ seconds to deallocate after a ~10 minute search. If more development time were available,
            // and ponder-based bot play were a priority, this would be one of the highest priorities for improvement.
            //
            // A workaround could be to delay post-ponder pruning until just after making a move, thereby eating into
            // ponder rather than search time. However, this risks running out of memory during the bot's turn.
            //
            // Another workaround could be to delegate pruning to a separate thread. Benefits aren't worth the technical risks/complications.

            timeControl.pondering = true;
            status = "pondering";

            // Add a hard limit of 5 minutes pondering to avoid running out of memory.
            timeControl.moveTimeMs = 5 * 60 * 1000;

            // Take the first-few-plies limit into account.
            if (limitSeconds)
            {
                timeControl.moveTimeMs = std::min(timeControl.moveTimeMs, static_cast<int64_t>(limitSeconds * 1000));
            }

            // Make sure that deallocation time doesn't eat into our search time too much
            // (conservatively, max 5% overhead, guess 10% deallocation time, 75% fudge on simplified remaining/increment).
            timeControl.moveTimeMs = std::min(
                timeControl.moveTimeMs,
                (3 * ((timeControl.timeRemainingMs[botSide] / Config::Misc.TimeControl_FractionOfRemaining) + timeControl.incrementMs[botSide]) / 8) - Config::Misc.TimeControl_SafetyBufferMoveMilliseconds);

            if (timeControl.moveTimeMs <= 0)
            {
                status = "waiting";
                skipSearch = true;
            }
        }

        // Even though it is more "correct" to capture the start time before pruning in "SearchUpdatePosition",
        // this can be dangerous with spiky overhead. An important move could be given near-zero time and blunder
        // the game away. We instead amortize the spiky overhead by setting an increased safety buffer in ChessCoachBot.cpp.
        Instance().workerGroup->searchState.Reset(timeControl, std::chrono::high_resolution_clock::now());
        Instance().workerGroup->searchState.botGameId = gameId;

        // Comment on the position and return the last SAN, as long as there's been at least one move,
        // since the encoder is trained with before-and-after positions (and as long as there's enough
        // think time remaining, and API throttling is under control).
        //
        // It's important to do this before waking up search workers so that this call can
        // return to Python quickly. Assume that the additional bot safety buffer covers the time needed.
        //
        // This acquires a "PythonContext" in the PythonNetwork call, so make it here in the
        // "NonPythonContext" so we that don't over-release (even though it's GIL-inefficient).
        const int botRemainingMilliseconds = ((botSide == WHITE) ? wtime : btime);
        if (!game.Moves().empty() && canComment && (botRemainingMilliseconds >= Config::Misc.Bot_CommentaryMinimumRemainingMilliseconds))
        {
            std::unique_ptr<INetwork::CommentaryInputPlanes> commentaryImage(new INetwork::CommentaryInputPlanes());
            game.GenerateCommentaryImage(commentaryImage->data());

            Position& position = game.GetPosition();
            const Move lastMove = game.Moves().back();
            position.undo_move(lastMove);
            san = Pgn::San(position, lastMove, true /* showCheckmate */);
            comments = Instance().network->PredictCommentaryBatch(1, commentaryImage.get());
        }

        // Start searching/pondering, but don't wait for workers.
        // Return to the bot loop in Python and let the primary search worker
        // send the best move asynchronously (like UCI "bestmove" to stdout).
        if (!skipSearch)
        {
            Instance().workerGroup->workCoordinator->ResetWorkItemsRemaining(1);
        }
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