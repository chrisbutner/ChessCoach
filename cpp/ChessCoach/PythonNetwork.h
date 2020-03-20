#ifndef _PYTHONNETWORK_H_
#define _PYTHONNETWORK_H_

#include "Network.h"

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

class PythonNetwork : public INetwork
{
public:

    static INetwork* GetLatestNetwork();

    PythonNetwork();
    virtual ~PythonNetwork();
    virtual IPrediction* Predict(InputPlanes& image) const;

private:

    PyObject* _predictModule;
    PyObject* _predictFunction;
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
    virtual IPrediction* Predict(InputPlanes& image) const;

private:

    // All-zero logits gives uniform softmax probability distribution.
    std::array<std::array<std::array<float, 8>, 8>, 73> _policy = {};
};

#endif // _PYTHONNETWORK_H_