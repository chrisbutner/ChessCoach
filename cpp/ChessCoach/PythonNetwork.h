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

class RawPrediction : public IPrediction
{
public:

    RawPrediction(float value, OutputPlanesPtr policy);

    virtual float Value() const;
    virtual void* Policy(); // To view as an OutputPlanesPtr

private:

    float _value;
    OutputPlanes _policy;
};

class BatchedPythonNetwork : public INetwork
{
public:

    static const int BatchSize =
#ifdef _DEBUG
        1;
#else
        16;
#endif

    BatchedPythonNetwork();
    virtual ~BatchedPythonNetwork();

    virtual void SetEnabled(bool enabled);
    virtual IPrediction* Predict(InputPlanes& image);
    virtual void TrainBatch(int step, InputPlanes* images, float* values, OutputPlanes* policies);
    virtual void SaveNetwork(int checkpoint);

    void Work();

private:

    bool _enabled;
    PyObject* _module;
    PyObject* _predictBatchFunction;
    PyObject* _trainBatchFunction;
    PyObject* _saveNetworkFunction;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::deque<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>> _predictQueue;
};

#endif // _PYTHONNETWORK_H_