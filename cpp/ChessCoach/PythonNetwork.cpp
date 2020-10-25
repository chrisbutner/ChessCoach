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
    PyAssert(sys);
    PyObject* sysPath = PyObject_GetAttrString(sys, "path");
    PyAssert(sysPath);
    PyObject* pythonPath = PyUnicode_FromString(Platform::InstallationScriptPath().string().c_str());
    PyAssert(pythonPath);
    const int error = PyList_Append(sysPath, pythonPath);
    PyAssert(!error);

    PyObject* module = PyImport_ImportModule("network");
    PyAssert(module);

    _predictBatchFunction[NetworkType_Teacher] = LoadFunction(module, "predict_batch_teacher");
    _predictBatchFunction[NetworkType_Student] = LoadFunction(module, "predict_batch_student");
    _predictCommentaryBatchFunction = LoadFunction(module, "predict_commentary_batch");
    _trainFunction[NetworkType_Teacher] = LoadFunction(module, "train_teacher");
    _trainFunction[NetworkType_Student] = LoadFunction(module, "train_student");
    _trainCommentaryFunction = LoadFunction(module, "train_commentary");
    _logScalarsFunction[NetworkType_Teacher] = LoadFunction(module, "log_scalars_teacher");
    _logScalarsFunction[NetworkType_Student] = LoadFunction(module, "log_scalars_student");
    _loadNetworkFunction = LoadFunction(module, "load_network");
    _saveNetworkFunction[NetworkType_Teacher] = LoadFunction(module, "save_network_teacher");
    _saveNetworkFunction[NetworkType_Student] = LoadFunction(module, "save_network_student");
    _saveFileFunction = LoadFunction(module, "save_file");

    Py_DECREF(module);
    Py_DECREF(pythonPath);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

PythonNetwork::~PythonNetwork()
{
    Py_XDECREF(_saveFileFunction);
    Py_XDECREF(_saveNetworkFunction[NetworkType_Student]);
    Py_XDECREF(_saveNetworkFunction[NetworkType_Teacher]);
    Py_XDECREF(_loadNetworkFunction);
    Py_XDECREF(_logScalarsFunction[NetworkType_Student]);
    Py_XDECREF(_logScalarsFunction[NetworkType_Teacher]);
    Py_XDECREF(_trainCommentaryFunction);
    Py_XDECREF(_trainFunction[NetworkType_Student]);
    Py_XDECREF(_trainFunction[NetworkType_Teacher]);
    Py_XDECREF(_predictCommentaryBatchFunction);
    Py_XDECREF(_predictBatchFunction[NetworkType_Student]);
    Py_XDECREF(_predictBatchFunction[NetworkType_Teacher]);
}

void PythonNetwork::PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[2]{ batchSize, InputPlaneCount };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_INT64, images);
    PyAssert(pythonImages);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictBatchFunction[networkType], pythonImages, nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonValues);
    PyAssert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = batchSize;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, values);

    // Network deals with tanh outputs/targets in (-1, 1)/[-1, 1]. MCTS deals with probabilities in [0, 1].
    MapProbabilities11To01(valueCount, values);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonPolicies);
    PyAssert(PyArray_Check(pythonPolicies));

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
    PyAssert(pythonImages);

    PyObject* result = PyObject_CallFunctionObjArgs(_predictCommentaryBatchFunction, pythonImages, nullptr);
    PyAssert(result);
    PyAssert(PyArray_Check(result));

    // Extract the comments.
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

void PythonNetwork::Train(NetworkType networkType, std::vector<GameType>& gameTypes,
    std::vector<Window>& trainingWindows, int step, int checkpoint)
{
    PythonContext context;

    npy_intp gameTypesDims[1]{ static_cast<int>(gameTypes.size()) };
    PyObject* pythonGameTypes = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(gameTypesDims), gameTypesDims, NPY_INT32, gameTypes.data());
    PyAssert(pythonGameTypes);

    const int windowClassIntFieldCount = 2;
    npy_intp trainingWindowsDims[2]{ static_cast<int>(trainingWindows.size()), windowClassIntFieldCount };
    PyObject* pythonTrainingWindows = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(trainingWindowsDims), trainingWindowsDims, NPY_INT32, trainingWindows.data());
    PyAssert(pythonTrainingWindows);

    PyObject* pythonStep = PyLong_FromLong(step);
    PyAssert(pythonStep);

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyAssert(pythonCheckpoint);

    PyObject* result = PyObject_CallFunctionObjArgs(_trainFunction[networkType], pythonGameTypes, pythonTrainingWindows,
        pythonStep, pythonCheckpoint, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonCheckpoint);
    Py_DECREF(pythonStep);
    Py_DECREF(pythonTrainingWindows);
    Py_DECREF(pythonGameTypes);
}

void PythonNetwork::TrainCommentary(int step, int checkpoint)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyAssert(pythonStep);

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyAssert(pythonCheckpoint);

    PyObject* result = PyObject_CallFunctionObjArgs(_trainCommentaryFunction, pythonStep, pythonCheckpoint, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonCheckpoint);
    Py_DECREF(pythonStep);
}

void PythonNetwork::LogScalars(NetworkType networkType, int step, int scalarCount, std::string* names, float* values)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyAssert(pythonStep);

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
    PyAssert(pythonNames);

    npy_intp valueDims[1]{ scalarCount };
    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(valueDims), valueDims, NPY_FLOAT32, values);
    PyAssert(pythonValues);

    PyObject* result = PyObject_CallFunctionObjArgs(_logScalarsFunction[networkType], pythonStep, pythonNames, pythonValues, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonNames);
    Py_DECREF(pythonStep);
}

void PythonNetwork::LoadNetwork(const std::string& networkName, int& stepCountOut, int& trainingChunkCountOut)
{
    PythonContext context;

    PyObject* pythonNetworkName = PyUnicode_FromStringAndSize(networkName.c_str(), networkName.size());
    PyAssert(pythonNetworkName);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_loadNetworkFunction, pythonNetworkName, nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract info.
    PyObject* pythonStepCount = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonStepCount);
    PyAssert(PyLong_Check(pythonStepCount));

    PyObject* pythonTrainingChunkCount = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonTrainingChunkCount);
    PyAssert(PyLong_Check(pythonTrainingChunkCount));

    stepCountOut = PyLong_AsLong(pythonStepCount);
    trainingChunkCountOut = PyLong_AsLong(pythonTrainingChunkCount);

    Py_DECREF(tupleResult);
    Py_DECREF(pythonNetworkName);
}

void PythonNetwork::SaveNetwork(NetworkType networkType, int checkpoint)
{
    PythonContext context;

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyAssert(pythonCheckpoint);

    PyObject* result = PyObject_CallFunctionObjArgs(_saveNetworkFunction[networkType], pythonCheckpoint, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonCheckpoint);
}

void PythonNetwork::SaveFile(const std::string& relativePath, const std::string& data)
{
    PythonContext context;

    PyObject* pythonRelativePath = PyUnicode_FromStringAndSize(relativePath.c_str(), relativePath.size());
    PyAssert(pythonRelativePath);

    PyObject* pythonData = PyBytes_FromStringAndSize(data.c_str(), data.size());
    PyAssert(pythonData);

    PyObject* result = PyObject_CallFunctionObjArgs(_saveFileFunction, pythonRelativePath, pythonData, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonData);
    Py_DECREF(pythonRelativePath);
}

PyObject* PythonNetwork::LoadFunction(PyObject* module, const char* name)
{
    PyObject* function = PyObject_GetAttrString(module, name);
    PyAssert(function);
    PyAssert(PyCallable_Check(function));
    return function;
}

void PythonNetwork::PyAssert(bool result)
{
    if (!result)
    {
        PyErr_Print();
    }
    assert(result);
}