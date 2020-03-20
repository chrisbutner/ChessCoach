#include "PythonNetwork.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PythonPrediction::PythonPrediction(PyObject* tupleResult)
    : _tupleResult(tupleResult)
{
    assert(tupleResult);

    // Extract the value.
    PyObject* pythonValue = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonValue));

    PyArrayObject* pythonValueArray = reinterpret_cast<PyArrayObject*>(pythonValue);
    _value = reinterpret_cast<float*>(PyArray_DATA(pythonValueArray))[0];

    // Extract the policy.
    PyObject* pythonPolicy = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonPolicy));

    PyArrayObject* pythonPolicyArray = reinterpret_cast<PyArrayObject*>(pythonPolicy);
    _policy = PyArray_DATA(pythonPolicyArray);
}

PythonPrediction::~PythonPrediction()
{
    Py_XDECREF(_tupleResult);
}

float PythonPrediction::Value() const
{
    return _value;
}

void* PythonPrediction::Policy() const
{
    return _policy;
}

INetwork* PythonNetwork::GetLatestNetwork()
{
    return new UniformNetwork();
}

PythonNetwork::PythonNetwork()
{
    if (!PyArray_API)
    {
        _import_array();
    }

    _predictModule = PyImport_ImportModule("predict");
    assert(_predictModule);

    _predictFunction = PyObject_GetAttrString(_predictModule, "predict");
    assert(_predictFunction);
    assert(PyCallable_Check(_predictFunction));
}

PythonNetwork::~PythonNetwork()
{
    Py_XDECREF(_predictFunction);
    Py_XDECREF(_predictModule);
}

IPrediction* PythonNetwork::Predict(InputPlanes& image) const
{
    PyObject* pythonImage = nullptr;

    npy_intp dims[3]{ 12, 8, 8 };

    pythonImage = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT, reinterpret_cast<void*>(image.data()));
    assert(pythonImage);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictFunction, pythonImage, nullptr);
    assert(tupleResult);

    Py_XDECREF(pythonImage);

    return new PythonPrediction(tupleResult);
}

UniformPrediction::UniformPrediction(void* policy)
    : _policy(policy)
{
}

UniformPrediction::~UniformPrediction()
{
}

float UniformPrediction::Value() const
{
    return 0.5f;
}

void* UniformPrediction::Policy() const
{
    return _policy;
}

UniformNetwork::UniformNetwork()
{
}

UniformNetwork::~UniformNetwork()
{
}

IPrediction* UniformNetwork::Predict(InputPlanes& image) const
{
    // "Promise" that Python won't modify the input tensor.
    UniformNetwork* that = const_cast<UniformNetwork*>(this);
    return new UniformPrediction(reinterpret_cast<void*>(that->_policy.data()));
}