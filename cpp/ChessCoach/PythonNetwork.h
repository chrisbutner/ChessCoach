#ifndef _PYTHONNETWORK_H_
#define _PYTHONNETWORK_H_

#include <vector>

#include "Network.h"
#include "Threading.h"

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

class PythonPrediction : public IPrediction
{
public:

    PythonPrediction(PyObject* tupleResult);
    virtual ~PythonPrediction();

    virtual float Value() const;
    virtual void* Policy() const; // To view as an OutputPlanesPtr

private:

    PyObject* _tupleResult;
    float _value;
    void* _policy;
};

class BatchedPythonPrediction : public IPrediction
{
public:

    BatchedPythonPrediction(PyObject* tupleResult, int index);
    virtual ~BatchedPythonPrediction();

    virtual float Value() const;
    virtual void* Policy() const; // To view as an OutputPlanesPtr

private:

    PyObject* _tupleResult;
    float _value;
    void* _policy;
};

class PythonNetwork : public INetwork
{
public:

    PythonNetwork();
    virtual ~PythonNetwork();
    virtual IPrediction* Predict(InputPlanes& image);

private:

    PyObject* _predictModule;
    PyObject* _predictFunction;
};

class BatchedPythonNetwork : public INetwork
{
public:

    static const int BatchSize = 16;

    BatchedPythonNetwork();
    virtual ~BatchedPythonNetwork();
    virtual IPrediction* Predict(InputPlanes& image);

private:

    PyObject* _predictModule;
    PyObject* _predictBatchFunction;
    std::mutex _mutex;
    std::vector<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>> _predictQueue;
};

class UniformPrediction : public IPrediction
{
public:

    UniformPrediction(void* policy);
    virtual ~UniformPrediction();

    virtual float Value() const;
    virtual void* Policy() const; // To view as an OutputPlanesPtr

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