#ifndef _PYTHONNETWORK_H_
#define _PYTHONNETWORK_H_

#include <vector>
#include <deque>

#include "Network.h"
#include "Threading.h"

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#define PY_ARRAY_UNIQUE_SYMBOL ChessCoach_ArrayApi
#define NO_IMPORT_ARRAY

class PythonContext
{
private:

    thread_local static PyGILState_STATE GilState;
    thread_local static PyThreadState* ThreadState;

public:

    PythonContext();
    ~PythonContext();
};

class PythonNetwork : public INetwork
{
public:

    static void PyAssert(bool result);

public:

    PythonNetwork();
    virtual ~PythonNetwork();

    virtual void PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies);
    virtual std::vector<std::string> PredictCommentaryBatch(int batchSize, InputPlanes* images);
    virtual void Train(NetworkType networkType, std::vector<GameType>& gameTypes,
        std::vector<Window>& trainingWindows, int step, int checkpoint);
    virtual void TrainCommentary(int step, int checkpoint);
    virtual void LogScalars(NetworkType networkType, int step, int scalarCount, std::string* names, float* values);
    virtual void LoadNetwork(const std::string& networkName, int& stepCountOut, int& trainingChunkCountOut);
    virtual void SaveNetwork(NetworkType networkType, int checkpoint);
    virtual void SaveFile(const std::string& relativePath, const std::string& data);

private:

    PyObject* LoadFunction(PyObject* module, const char* name);

private:

    PyObject* _predictBatchFunction[NetworkType_Count];
    PyObject* _predictCommentaryBatchFunction;
    PyObject* _trainFunction[NetworkType_Count];
    PyObject* _trainCommentaryFunction;
    PyObject* _logScalarsFunction[NetworkType_Count];
    PyObject* _loadNetworkFunction;
    PyObject* _saveNetworkFunction[NetworkType_Count];
    PyObject* _saveFileFunction;
};

#endif // _PYTHONNETWORK_H_