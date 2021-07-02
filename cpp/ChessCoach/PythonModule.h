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

#ifndef _PYTHONMODULE_H_
#define _PYTHONMODULE_H_

#include <string>

#include "PythonNetwork.h"
#include "Storage.h"
#include "WorkerGroup.h"

class PythonModule
{
public:

    static PyObject* PyInit_ChessCoachModule();
    static PyMethodDef ChessCoachMethods[];
    static PyModuleDef ChessCoachModule;

    static PythonModule& Instance();

private:

    static PyObject* LoadChunk(PyObject* self, PyObject* args);
    static PyObject* LoadGame(PyObject* self, PyObject* args);
    static PyObject* LoadPosition(PyObject* self, PyObject* args);
    static PyObject* ShowLine(PyObject* self, PyObject* args);
    static PyObject* EvaluateParameters(PyObject* self, PyObject* args);
    static PyObject* GenerateCommentaryImageForFens(PyObject* self, PyObject* args);
    static PyObject* GenerateCommentaryImageForPosition(PyObject* self, PyObject* args);
    static PyObject* BotSearch(PyObject* self, PyObject* args);

public:

    INetwork* network = nullptr;
    Storage* storage = nullptr;
    WorkerGroup* workerGroup = nullptr;

private:

    std::string _chunkContents;
    SavedGame _game;
};

#endif // _PYTHONMODULE_H_