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

    static const int BatchSize = 1;

    BatchedPythonNetwork();
    virtual ~BatchedPythonNetwork();

    virtual IPrediction* Predict(InputPlanes& image);

    void Work();

private:

    PyObject* _module;
    PyObject* _predictBatchFunction;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::deque<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>> _predictQueue;
};

class UniformPrediction : public IPrediction
{
public:

    UniformPrediction(void* policy);
    virtual ~UniformPrediction();

    virtual float Value() const;
    virtual void* Policy(); // To view as an OutputPlanesPtr

private:

    void* _policy;
};

class UniformNetwork : public INetwork
{
public:

    UniformNetwork();
    virtual ~UniformNetwork();

    virtual IPrediction* Predict(InputPlanes& image);

private:

    // All-zero logits gives uniform softmax probability distribution.
    std::array<std::array<std::array<float, 8>, 8>, 73> _policy = {};
};

#endif // _PYTHONNETWORK_H_