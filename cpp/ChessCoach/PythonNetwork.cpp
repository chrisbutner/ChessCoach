#include "PythonNetwork.h"

#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

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
    PyCallAssert(_module);

    _predictBatchFunction = PyObject_GetAttrString(_module, "predict_batch");
    PyCallAssert(_predictBatchFunction);
    PyCallAssert(PyCallable_Check(_predictBatchFunction));

    _trainBatchFunction = PyObject_GetAttrString(_module, "train_batch");
    PyCallAssert(_trainBatchFunction);
    PyCallAssert(PyCallable_Check(_trainBatchFunction));

    _testBatchFunction = PyObject_GetAttrString(_module, "test_batch");
    PyCallAssert(_testBatchFunction);
    PyCallAssert(PyCallable_Check(_testBatchFunction));

    _logScalarsFunction = PyObject_GetAttrString(_module, "log_scalars");
    PyCallAssert(_logScalarsFunction);
    PyCallAssert(PyCallable_Check(_logScalarsFunction));

    _saveNetworkFunction = PyObject_GetAttrString(_module, "save_network");
    PyCallAssert(_saveNetworkFunction);
    PyCallAssert(PyCallable_Check(_saveNetworkFunction));
}

BatchedPythonNetwork::~BatchedPythonNetwork()
{
    Py_XDECREF(_saveNetworkFunction);
    Py_XDECREF(_logScalarsFunction);
    Py_XDECREF(_trainBatchFunction);
    Py_XDECREF(_testBatchFunction);
    Py_XDECREF(_predictBatchFunction);
    Py_XDECREF(_module);
}

void BatchedPythonNetwork::PredictBatch(int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[4]{ batchSize, InputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_FLOAT32, images);
    PyCallAssert(pythonImages);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictBatchFunction, pythonImages, nullptr);
    PyCallAssert(tupleResult);

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyCallAssert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = batchSize;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, values);

    // Network deals with tanh outputs/targets in (-1, 1)/[-1, 1]. MCTS deals with probabilities in [0, 1].
    MapProbabilities11To01(valueCount, values);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyCallAssert(PyArray_Check(pythonPolicies));

    PyArrayObject* pythonPoliciesArray = reinterpret_cast<PyArrayObject*>(pythonPolicies);
    float* pythonPoliciesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonPoliciesArray));

    const int policyCount = (batchSize * OutputPlanesFloatCount);
    std::copy(pythonPoliciesPtr, pythonPoliciesPtr + policyCount, reinterpret_cast<float*>(policies));

    Py_DECREF(tupleResult);
    Py_DECREF(pythonImages);
}

void BatchedPythonNetwork::TrainTestBatch(PyObject* function, int step, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    // MCTS deals with probabilities in [0, 1]. Network deals with tanh outputs/targets in (-1, 1)/[-1, 1].
    MapProbabilities01To11(batchSize, values);

    PyObject* pythonStep = PyLong_FromLong(step);
    PyCallAssert(pythonStep);

    npy_intp imageDims[4]{ batchSize, InputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_FLOAT32, images);
    PyCallAssert(pythonImages);

    npy_intp valueDims[1]{ batchSize };
    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, values);
    PyCallAssert(pythonValues);

    npy_intp policyDims[4]{ batchSize, OutputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonPolicies = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, policies);
    PyCallAssert(pythonPolicies);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(function, pythonStep, pythonImages, pythonValues, pythonPolicies, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonPolicies);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonImages);
    Py_DECREF(pythonStep);
}

void BatchedPythonNetwork::TrainBatch(int step, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    TrainTestBatch(_trainBatchFunction, step, batchSize, images, values, policies);
}

void BatchedPythonNetwork::TestBatch(int step, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    TrainTestBatch(_testBatchFunction, step, batchSize, images, values, policies);
}

void BatchedPythonNetwork::LogScalars(int step, int scalarCount, std::string* names, float* values)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyCallAssert(pythonStep);

    // Pack the strings contiguously.
    int longestString = 0;
    for (int i = 0; i < scalarCount; i++)
    {
        longestString = std::max(longestString, static_cast<int>(names[i].size()));
    }
    void* memory = PyDataMem_NEW(longestString * scalarCount);
    assert(memory);
    char* packedStrings = reinterpret_cast<char*>(memory);
    for (int i = 0; i < scalarCount; i++)
    {
        char* packedString = (packedStrings + (i * longestString));
        std::copy(&names[i][0], &names[i][0] + names[i].size(), packedString);
        std::fill(packedString + names[i].size(), packedString + longestString, '\0');
    }

    npy_intp nameDims[1]{ scalarCount };
    PyObject* pythonNames = PyArray_New(&PyArray_Type, Py_ARRAY_LENGTH(nameDims), nameDims,
        NPY_STRING, nullptr, memory, longestString, NPY_ARRAY_OWNDATA, nullptr);
    PyCallAssert(pythonNames);

    npy_intp valueDims[1]{ scalarCount };
    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, values);
    PyCallAssert(pythonValues);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_logScalarsFunction, pythonStep, pythonNames, pythonValues, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonNames);
    Py_DECREF(pythonStep);
}

void BatchedPythonNetwork::SaveNetwork(int checkpoint)
{
    PythonContext context;

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyCallAssert(pythonCheckpoint);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_saveNetworkFunction, pythonCheckpoint, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonCheckpoint);
}

void BatchedPythonNetwork::PyCallAssert(bool result)
{
    if (!result)
    {
        PyErr_Print();
    }
    assert(result);
}