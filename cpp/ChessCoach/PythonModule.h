#ifndef _PYTHONMODULE_H_
#define _PYTHONMODULE_H_

#include <string>

#include "PythonNetwork.h"
#include "Storage.h"

class PythonModule
{
public:

    static PyObject* PyInit_ChessCoachModule();
    static PyMethodDef ChessCoachMethods[];
    static PyModuleDef ChessCoachModule;

private:

    static PyObject* LoadChunk(PyObject* self, PyObject* args);
    static PyObject* LoadGame(PyObject* self, PyObject* args);
    static PyObject* LoadPosition(PyObject* self, PyObject* args);

public:

    PythonModule();

private:

    static PythonModule& Instance();

private:

    Storage _storage;
    std::string _chunkContents;
    SavedGame _game;
};

#endif // _PYTHONMODULE_H_