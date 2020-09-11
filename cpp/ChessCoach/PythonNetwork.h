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

    PythonNetwork();
    virtual ~PythonNetwork();

    virtual void PredictBatch(int batchSize, InputPlanes* images, float* values, OutputPlanes* policies);
    virtual std::vector<std::string> PredictCommentaryBatch(int batchSize, InputPlanes* images);
    virtual void TrainBatch(int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
        OutputPlanes* policies, OutputPlanes* replyPolicies);
    virtual void ValidateBatch(int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
        OutputPlanes* policies, OutputPlanes* replyPolicies);
    virtual void TrainCommentaryBatch(int step, int batchSize, InputPlanes* images, std::string* comments);
    virtual void LogScalars(int step, int scalarCount, std::string* names, float* values);
    virtual void LoadNetwork(const char* networkName);
    virtual void SaveNetwork(int checkpoint);

private:

    void TrainValidateBatch(PyObject* function, int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
        OutputPlanes* policies, OutputPlanes* replyPolicies);
    PyObject* LoadFunction(PyObject* module, const char* name);
    void PyCallAssert(bool result);

private:

    PyObject* _predictBatchFunction;
    PyObject* _predictCommentaryBatchFunction;
    PyObject* _trainBatchFunction;
    PyObject* _validateBatchFunction;
    PyObject* _trainCommentaryBatchFunction;
    PyObject* _logScalarsFunction;
    PyObject* _loadNetworkFunction;
    PyObject* _saveNetworkFunction;
};

#endif // _PYTHONNETWORK_H_