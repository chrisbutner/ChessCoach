#ifndef _PYTHONMODULE_H_
#define _PYTHONMODULE_H_

#include <string>

#include "PythonNetwork.h"
#include "Storage.h"
#include "SelfPlay.h"

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
    static PyObject* EvaluateParameters(PyObject* self, PyObject* args);

public:

    SelfPlayWorker* worker = nullptr;
    INetwork* network = nullptr;
    Storage* storage = nullptr;

private:

    std::string _chunkContents;
    SavedGame _game;
};

#endif // _PYTHONMODULE_H_