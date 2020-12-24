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

NonPythonContext::NonPythonContext()
{
    // Release the GIL.
    _threadState = PyEval_SaveThread();
}

NonPythonContext::~NonPythonContext()
{
    // Re-acquire the GIL.
    PyEval_RestoreThread(_threadState);
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
    _getNetworkInfoFunction[NetworkType_Teacher] = LoadFunction(module, "get_network_info_teacher");
    _getNetworkInfoFunction[NetworkType_Student] = LoadFunction(module, "get_network_info_student");
    _saveFileFunction = LoadFunction(module, "save_file");
    _loadFileFunction = LoadFunction(module, "load_file");
    _fileExistsFunction = LoadFunction(module, "file_exists");
    _launchGuiFunction = LoadFunction(module, "launch_gui");
    _updateGuiFunction = LoadFunction(module, "update_gui");
    _debugDecompressFunction = LoadFunction(module, "debug_decompress");

    Py_DECREF(module);
    Py_DECREF(pythonPath);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

PythonNetwork::~PythonNetwork()
{
    Py_XDECREF(_debugDecompressFunction);
    Py_XDECREF(_fileExistsFunction);
    Py_XDECREF(_updateGuiFunction);
    Py_XDECREF(_launchGuiFunction);
    Py_XDECREF(_loadFileFunction);
    Py_XDECREF(_saveFileFunction);
    Py_XDECREF(_getNetworkInfoFunction[NetworkType_Student]);
    Py_XDECREF(_getNetworkInfoFunction[NetworkType_Teacher]);
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

PredictionStatus PythonNetwork::PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies)
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

    // Get the prediction status.
    PyObject* pythonPredictionStatus = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonPredictionStatus);
    PyAssert(PyLong_Check(pythonPredictionStatus));
    const PredictionStatus status = static_cast<PredictionStatus>(PyLong_AsLong(pythonPredictionStatus));

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonValues);
    PyAssert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = batchSize;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, values);

    // Network deals with tanh outputs/targets in (-1, 1)/[-1, 1]. MCTS deals with probabilities in [0, 1].
    MapProbabilities11To01(valueCount, values);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 2); // PyTuple_GetItem does not INCREF
    PyAssert(pythonPolicies);
    PyAssert(PyArray_Check(pythonPolicies));

    PyArrayObject* pythonPoliciesArray = reinterpret_cast<PyArrayObject*>(pythonPolicies);
    PlanesPointerFlat pythonPoliciesPtr = reinterpret_cast<PlanesPointerFlat>(PyArray_DATA(pythonPoliciesArray));

    const int policyCount = (batchSize * OutputPlanesFloatCount);
    std::copy(pythonPoliciesPtr, pythonPoliciesPtr + policyCount, reinterpret_cast<PlanesPointerFlat>(policies));

    Py_DECREF(tupleResult);
    Py_DECREF(pythonImages);

    return status;
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

void PythonNetwork::LogScalars(NetworkType networkType, int step, const std::vector<std::string> names, float* values)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyAssert(pythonStep);

    PyObject* pythonNames = PackNumpyStringArray(names);

    npy_intp valueDims[1]{ static_cast<int>(names.size()) };
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

void PythonNetwork::LoadNetwork(const std::string& networkName)
{
    PythonContext context;

    PyObject* pythonNetworkName = PyUnicode_FromStringAndSize(networkName.c_str(), networkName.size());
    PyAssert(pythonNetworkName);

    PyObject* result = PyObject_CallFunctionObjArgs(_loadNetworkFunction, pythonNetworkName, nullptr);
    PyAssert(result);

    Py_DECREF(result);
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

void PythonNetwork::GetNetworkInfo(NetworkType networkType, int* stepCountOut, int* trainingChunkCountOut, std::string* relativePathOut)
{
    PythonContext context;

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_getNetworkInfoFunction[networkType], nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract info.
    PyObject* pythonStepCount = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonStepCount);
    PyAssert(PyLong_Check(pythonStepCount));

    PyObject* pythonTrainingChunkCount = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonTrainingChunkCount);
    PyAssert(PyLong_Check(pythonTrainingChunkCount));

    PyObject* pythonRelativePath = PyTuple_GetItem(tupleResult, 2); // PyTuple_GetItem does not INCREF
    PyAssert(pythonRelativePath);
    PyAssert(PyBytes_Check(pythonRelativePath));

    if (stepCountOut)
    {
        *stepCountOut = PyLong_AsLong(pythonStepCount);
    }
    if (trainingChunkCountOut)
    {
        *trainingChunkCountOut = PyLong_AsLong(pythonTrainingChunkCount);
    }
    if (relativePathOut)
    {
        *relativePathOut = PyBytes_AsString(pythonRelativePath);
    }

    Py_DECREF(tupleResult);
}

void PythonNetwork::SaveFile(const std::string& relativePath, const std::string& data)
{
    PythonContext context;

    PyObject* pythonRelativePath = PyUnicode_FromStringAndSize(relativePath.data(), relativePath.size());
    PyAssert(pythonRelativePath);

    PyObject* pythonData = PyBytes_FromStringAndSize(data.data(), data.size());
    PyAssert(pythonData);

    PyObject* result = PyObject_CallFunctionObjArgs(_saveFileFunction, pythonRelativePath, pythonData, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonData);
    Py_DECREF(pythonRelativePath);
}

std::string PythonNetwork::LoadFile(const std::string& relativePath)
{
    PythonContext context;

    PyObject* pythonRelativePath = PyUnicode_FromStringAndSize(relativePath.data(), relativePath.size());
    PyAssert(pythonRelativePath);

    PyObject* result = PyObject_CallFunctionObjArgs(_loadFileFunction, pythonRelativePath, nullptr);
    PyAssert(result);
    PyAssert(PyBytes_Check(result));

    char* bytes;
    Py_ssize_t size;
    PyBytes_AsStringAndSize(result, &bytes, &size);
    std::string fileData(bytes, size);

    Py_DECREF(result);
    Py_DECREF(pythonRelativePath);

    return fileData;
}

bool PythonNetwork::FileExists(const std::string& relativePath)
{
    PythonContext context;

    PyObject* pythonRelativePath = PyUnicode_FromStringAndSize(relativePath.data(), relativePath.size());
    PyAssert(pythonRelativePath);

    PyObject* result = PyObject_CallFunctionObjArgs(_fileExistsFunction, pythonRelativePath, nullptr);
    PyAssert(result);
    PyAssert(PyBool_Check(result));
    const bool exists = PyObject_IsTrue(result);

    Py_DECREF(result);
    Py_DECREF(pythonRelativePath);

    return exists;
}

void PythonNetwork::LaunchGui(const std::string& mode)
{
    PythonContext context;

    PyObject* pythonMode = PyUnicode_FromStringAndSize(mode.data(), mode.size());
    PyAssert(pythonMode);

    PyObject* result = PyObject_CallFunctionObjArgs(_launchGuiFunction, pythonMode, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonMode);
}

void PythonNetwork::UpdateGui(const std::string& fen, int nodeCount, const std::string& evaluation, const std::string& principleVariation,
    const std::vector<std::string>& sans, const std::vector<std::string>& froms, const std::vector<std::string>& tos, std::vector<float>& policyValues)
{
    PythonContext context;

    PyObject* pythonFen = PyUnicode_FromStringAndSize(fen.data(), fen.size());
    PyAssert(pythonFen);

    PyObject* pythonNodeCount = PyLong_FromLong(nodeCount);
    PyAssert(pythonNodeCount);

    PyObject* pythonEvaluation = PyUnicode_FromStringAndSize(evaluation.data(), evaluation.size());
    PyAssert(pythonEvaluation);

    PyObject* pythonPrincipleVariation = PyUnicode_FromStringAndSize(principleVariation.data(), principleVariation.size());
    PyAssert(pythonPrincipleVariation);

    PyObject* pythonSans = PackNumpyStringArray(sans);
    PyObject* pythonFroms = PackNumpyStringArray(froms);
    PyObject* pythonTos = PackNumpyStringArray(tos);

    npy_intp policyValueDims[1]{ static_cast<int>(policyValues.size()) };
    PyObject* pythonPolicyValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyValueDims), policyValueDims, NPY_FLOAT32, policyValues.data());
    PythonNetwork::PyAssert(pythonPolicyValues);

    PyObject* result = PyObject_CallFunctionObjArgs(_updateGuiFunction, pythonFen, pythonNodeCount, pythonEvaluation, pythonPrincipleVariation,
        pythonSans, pythonFroms, pythonTos, pythonPolicyValues, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonPolicyValues);
    Py_DECREF(pythonTos);
    Py_DECREF(pythonFroms);
    Py_DECREF(pythonSans);
    Py_DECREF(pythonPrincipleVariation);
    Py_DECREF(pythonEvaluation);
    Py_DECREF(pythonNodeCount);
    Py_DECREF(pythonFen);
}

void PythonNetwork::DebugDecompress(int positionCount, int policySize, float* result, int64_t* imagePiecesAuxiliary,
    float* mctsValues, int64_t* policyRowLengths, int64_t* policyIndices, float* policyValues, InputPlanes* imagesOut,
    float* valuesOut, OutputPlanes* policiesOut)
{
    PythonContext context;

    // Compressed probabilities are already in Python/TensorFlow [-1, 1] range (see Storage::PopulateGame).

    // Wrap compressed data in numpy.
    PyObject* pythonResult = PyArray_SimpleNewFromData(0, 0, NPY_FLOAT32, result);
    PyAssert(pythonResult);

    npy_intp imagePiecesAuxiliaryDims[2]{ positionCount, InputPiecePlanesPerPosition + InputAuxiliaryPlaneCount };
    PyObject* pythonImagePiecesAuxiliary = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imagePiecesAuxiliaryDims), imagePiecesAuxiliaryDims, NPY_INT64, imagePiecesAuxiliary);
    PyAssert(pythonImagePiecesAuxiliary);

    npy_intp perPositionDims[1]{ positionCount };
    PyObject* pythonMctsValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(perPositionDims), perPositionDims, NPY_FLOAT32, mctsValues);
    PyAssert(pythonMctsValues);

    PyObject* pythonPolicyRowLengths = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(perPositionDims), perPositionDims, NPY_INT64, policyRowLengths);
    PyAssert(pythonPolicyRowLengths);

    npy_intp policyDims[1]{ policySize };
    PyObject* pythonPolicyIndices = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_INT64, policyIndices);
    PyAssert(pythonPolicyIndices);

    PyObject* pythonPolicyValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, policyValues);
    PyAssert(pythonPolicyValues);

    // Make the call.
    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_debugDecompressFunction, pythonResult, pythonImagePiecesAuxiliary,
        pythonMctsValues, pythonPolicyRowLengths, pythonPolicyIndices, pythonPolicyValues, nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract the images.
    PyObject* pythonImages = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonImages);
    PyAssert(PyArray_Check(pythonImages));

    PyArrayObject* pythonImagesArray = reinterpret_cast<PyArrayObject*>(pythonImages);
    PackedPlane* pythonImagesPtr = reinterpret_cast<PackedPlane*>(PyArray_DATA(pythonImagesArray));

    const int imageCount = (positionCount * InputPlaneCount);
    std::copy(pythonImagesPtr, pythonImagesPtr + imageCount, reinterpret_cast<PackedPlane*>(imagesOut));

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonValues);
    PyAssert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = positionCount;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, valuesOut);

    // Network deals with tanh outputs/targets in (-1, 1)/[-1, 1]. MCTS deals with probabilities in [0, 1].
    MapProbabilities11To01(valueCount, valuesOut);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 2); // PyTuple_GetItem does not INCREF
    PyAssert(pythonPolicies);
    PyAssert(PyArray_Check(pythonPolicies));

    PyArrayObject* pythonPoliciesArray = reinterpret_cast<PyArrayObject*>(pythonPolicies);
    PlanesPointerFlat pythonPoliciesPtr = reinterpret_cast<PlanesPointerFlat>(PyArray_DATA(pythonPoliciesArray));

    const int policyCount = (positionCount * OutputPlanesFloatCount);
    std::copy(pythonPoliciesPtr, pythonPoliciesPtr + policyCount, reinterpret_cast<PlanesPointerFlat>(policiesOut));

    Py_DECREF(tupleResult);
    Py_DECREF(pythonPolicyValues);
    Py_DECREF(pythonPolicyIndices);
    Py_DECREF(pythonPolicyRowLengths);
    Py_DECREF(pythonMctsValues);
    Py_DECREF(pythonImagePiecesAuxiliary);
    Py_DECREF(pythonResult);
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

PyObject* PythonNetwork::PackNumpyStringArray(const std::vector<std::string>& values)
{
    // Pack the strings contiguously.
    const int valueCount = static_cast<int>(values.size());
    int longestString = 0;
    for (const std::string& value : values)
    {
        longestString = std::max(longestString, static_cast<int>(value.length()));
    }
    void* memory = PyDataMem_NEW(longestString * valueCount);
    PyAssert(memory);
    char* packedStrings = reinterpret_cast<char*>(memory);
    for (int i = 0; i < valueCount; i++)
    {
        char* packedString = (packedStrings + (i * longestString));
        std::copy(&values[i][0], &values[i][0] + values[i].length(), packedString);
        std::fill(packedString + values[i].length(), packedString + longestString, '\0');
    }

    npy_intp dims[1]{ valueCount };
    PyObject* pythonValues = PyArray_New(&PyArray_Type, Py_ARRAY_LENGTH(dims), dims,
        NPY_STRING, nullptr, memory, longestString, NPY_ARRAY_OWNDATA, nullptr);
    PyAssert(pythonValues);
    return pythonValues;
}