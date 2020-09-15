#include "PythonNetwork.h"

#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "Platform.h"

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

PythonNetwork::PythonNetwork()
{
    PythonContext context;

    PyObject* sys = PyImport_ImportModule("sys");
    PyCallAssert(sys);
    PyObject* sysPath = PyObject_GetAttrString(sys, "path");
    PyCallAssert(sysPath);
    PyObject* pythonPath = PyUnicode_FromString(Platform::InstallationScriptPath().string().c_str());
    PyCallAssert(pythonPath);
    const int error = PyList_Append(sysPath, pythonPath);
    PyCallAssert(!error);

    PyObject* module = PyImport_ImportModule("network");
    PyCallAssert(module);

    _predictBatchFunction = LoadFunction(module, "predict_batch");
    _predictCommentaryBatchFunction = LoadFunction(module, "predict_commentary_batch");
    _trainBatchFunction = LoadFunction(module, "train_batch");
    _validateBatchFunction = LoadFunction(module, "validate_batch");
    _trainCommentaryBatchFunction = LoadFunction(module, "train_commentary_batch");
    _logScalarsFunction = LoadFunction(module, "log_scalars");
    _loadNetworkFunction = LoadFunction(module, "load_network");
    _saveNetworkFunction = LoadFunction(module, "save_network");

    Py_DECREF(module);
    Py_DECREF(pythonPath);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

PythonNetwork::~PythonNetwork()
{
    Py_XDECREF(_saveNetworkFunction);
    Py_XDECREF(_loadNetworkFunction);
    Py_XDECREF(_logScalarsFunction);
    Py_XDECREF(_trainCommentaryBatchFunction);
    Py_XDECREF(_validateBatchFunction);
    Py_XDECREF(_trainBatchFunction);
    Py_XDECREF(_predictCommentaryBatchFunction);
    Py_XDECREF(_predictBatchFunction);
}

void PythonNetwork::PredictBatch(int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[2]{ batchSize, InputPlaneCount };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64, images);
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

std::vector<std::string> PythonNetwork::PredictCommentaryBatch(int batchSize, InputPlanes* images)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[2]{ batchSize, InputPlaneCount };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64, images);
    PyCallAssert(pythonImages);

    PyObject* result = PyObject_CallFunctionObjArgs(_predictCommentaryBatchFunction, pythonImages, nullptr);
    PyCallAssert(result);
    PyCallAssert(PyArray_Check(result));

    // Extract the values.
    PyArrayObject* pythonCommentsArray = reinterpret_cast<PyArrayObject*>(result);

    std::vector<std::string> comments(batchSize);
    for (int i = 0; i < batchSize; i++)
    {
        comments[i] = reinterpret_cast<const char*>(PyArray_GETPTR1(pythonCommentsArray, i));
    }

    Py_DECREF(result);
    Py_DECREF(pythonImages);

    return comments;
}

void PythonNetwork::TrainValidateBatch(PyObject* function, int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
    OutputPlanes* policies, OutputPlanes* replyPolicies)
{
    PythonContext context;

    // MCTS deals with probabilities in [0, 1]. Network deals with tanh outputs/targets in (-1, 1)/[-1, 1].
    MapProbabilities01To11(batchSize, values);
    MapProbabilities01To11(batchSize, mctsValues);

    PyObject* pythonStep = PyLong_FromLong(step);
    PyCallAssert(pythonStep);

    npy_intp imageDims[2]{ batchSize, InputPlaneCount };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64, images);
    PyCallAssert(pythonImages);

    npy_intp valueDims[1]{ batchSize };
    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, values);
    PyCallAssert(pythonValues);

    PyObject* pythonMctsValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, mctsValues);
    PyCallAssert(pythonMctsValues);

    npy_intp policyDims[4]{ batchSize, OutputPlaneCount, BoardSide, BoardSide };
    PyObject* pythonPolicies = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, policies);
    PyCallAssert(pythonPolicies);

    PyObject* pythonReplyPolicies = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, replyPolicies);
    PyCallAssert(pythonReplyPolicies);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(function, pythonStep, pythonImages,
        pythonValues, pythonMctsValues, pythonPolicies, pythonReplyPolicies, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonReplyPolicies);
    Py_DECREF(pythonPolicies);
    Py_DECREF(pythonMctsValues);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonImages);
    Py_DECREF(pythonStep);
}

void PythonNetwork::TrainBatch(int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
    OutputPlanes* policies, OutputPlanes* replyPolicies)
{
    TrainValidateBatch(_trainBatchFunction, step, batchSize, images, values, mctsValues, policies, replyPolicies);
}

void PythonNetwork::ValidateBatch(int step, int batchSize, InputPlanes* images, float* values, float* mctsValues,
    OutputPlanes* policies, OutputPlanes* replyPolicies)
{
    TrainValidateBatch(_validateBatchFunction, step, batchSize, images, values, mctsValues, policies, replyPolicies);
}

void PythonNetwork::TrainCommentaryBatch(int step, int batchSize, InputPlanes* images, std::string* comments)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyCallAssert(pythonStep);

    npy_intp imageDims[2]{ batchSize, InputPlaneCount };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64, images);
    PyCallAssert(pythonImages);

    // Pack the strings contiguously.
    int longestString = 0;
    for (int i = 0; i < batchSize; i++)
    {
        longestString = std::max(longestString, static_cast<int>(comments[i].size()));
    }
    void* memory = PyDataMem_NEW(longestString * batchSize);
    assert(memory);
    char* packedStrings = reinterpret_cast<char*>(memory);
    for (int i = 0; i < batchSize; i++)
    {
        char* packedString = (packedStrings + (i * longestString));
        std::copy(&comments[i][0], &comments[i][0] + comments[i].size(), packedString);
        std::fill(packedString + comments[i].size(), packedString + longestString, '\0');
    }

    npy_intp commentDims[1]{ batchSize };
    PyObject* pythonComments = PyArray_New(&PyArray_Type, Py_ARRAY_LENGTH(commentDims), commentDims,
        NPY_STRING, nullptr, memory, longestString, NPY_ARRAY_OWNDATA, nullptr);
    PyCallAssert(pythonComments);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_trainCommentaryBatchFunction, pythonStep, pythonImages,
        pythonComments, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonComments);
    Py_DECREF(pythonImages);
    Py_DECREF(pythonStep);

}

void PythonNetwork::LogScalars(int step, int scalarCount, std::string* names, float* values)
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

void PythonNetwork::LoadNetwork(const char* networkName)
{
    PythonContext context;

    PyObject* pythonNetworkName = PyUnicode_FromString(networkName);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_loadNetworkFunction, pythonNetworkName, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonNetworkName);
}

void PythonNetwork::SaveNetwork(int checkpoint)
{
    PythonContext context;

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyCallAssert(pythonCheckpoint);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_saveNetworkFunction, pythonCheckpoint, nullptr);
    PyCallAssert(tupleResult);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonCheckpoint);
}

PyObject* PythonNetwork::LoadFunction(PyObject* module, const char* name)
{
    PyObject* function = PyObject_GetAttrString(module, name);
    PyCallAssert(function);
    PyCallAssert(PyCallable_Check(function));
    return function;
}

void PythonNetwork::PyCallAssert(bool result)
{
    if (!result)
    {
        PyErr_Print();
    }
    assert(result);
}