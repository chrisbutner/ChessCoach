#include "PythonNetwork.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Config.h"

thread_local PyGILState_STATE PythonContext::GilState;
thread_local PyThreadState* PythonContext::ThreadState = nullptr;

PythonContext::PythonContext()
{
    // Re-acquire the GIL.
    if (!ThreadState)
    {
        GilState = PyGILState_Ensure();
    }
    else
    {
        PyEval_RestoreThread(ThreadState);
    }

    // Ensure numpy is initialized.
    if (!PyArray_API)
    {
        _import_array();
    }
}

PythonContext::~PythonContext()
{
    // Release the GIL.
    ThreadState = PyEval_SaveThread();
}

BatchedPythonNetwork::BatchedPythonNetwork()
{
    PythonContext context;

    _module = PyImport_ImportModule("network");
    assert(_module);

    _predictBatchFunction = PyObject_GetAttrString(_module, "predict_batch");
    assert(_predictBatchFunction);
    assert(PyCallable_Check(_predictBatchFunction));

    _trainBatchFunction = PyObject_GetAttrString(_module, "train_batch");
    assert(_trainBatchFunction);
    assert(PyCallable_Check(_trainBatchFunction));

    _saveNetworkFunction = PyObject_GetAttrString(_module, "save_network");
    assert(_saveNetworkFunction);
    assert(PyCallable_Check(_saveNetworkFunction));
}

BatchedPythonNetwork::~BatchedPythonNetwork()
{
    Py_XDECREF(_saveNetworkFunction);
    Py_XDECREF(_trainBatchFunction);
    Py_XDECREF(_predictBatchFunction);
    Py_XDECREF(_module);
}

void BatchedPythonNetwork::PredictBatch(InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[4]{ Config::PredictionBatchSize, InputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_FLOAT32, images);
    assert(pythonImages);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictBatchFunction, pythonImages, nullptr);
    assert(tupleResult);

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = Config::PredictionBatchSize;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, values);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonPolicies));

    PyArrayObject* pythonPoliciesArray = reinterpret_cast<PyArrayObject*>(pythonPolicies);
    float* pythonPoliciesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonPoliciesArray));

    const int policyCount = (Config::PredictionBatchSize * OutputPlanesFloatCount);
    std::copy(pythonPoliciesPtr, pythonPoliciesPtr + policyCount, reinterpret_cast<float*>(policies));

    Py_DECREF(tupleResult);
    Py_DECREF(pythonImages);
}

void BatchedPythonNetwork::TrainBatch(int step, InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    assert(pythonStep);

    npy_intp imageDims[4]{ Config::BatchSize, InputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_FLOAT32, images);
    assert(pythonImages);

    npy_intp valueDims[1]{ Config::BatchSize };
    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, values);
    assert(pythonValues);

    npy_intp policyDims[4]{ Config::BatchSize, OutputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonPolicies = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, policies);
    assert(pythonPolicies);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_trainBatchFunction, pythonStep, pythonImages, pythonValues, pythonPolicies, nullptr);
    assert(tupleResult);

    Py_XDECREF(tupleResult);
    Py_DECREF(pythonPolicies);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonImages);
    Py_DECREF(pythonStep);
}

void BatchedPythonNetwork::SaveNetwork(int checkpoint)
{
    PythonContext context;

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    assert(pythonCheckpoint);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_saveNetworkFunction, pythonCheckpoint, nullptr);
    assert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonCheckpoint);
}