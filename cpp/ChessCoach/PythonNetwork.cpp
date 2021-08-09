// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

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
    _saveNetworkFunction[NetworkType_Teacher] = LoadFunction(module, "save_network_teacher");
    _saveNetworkFunction[NetworkType_Student] = LoadFunction(module, "save_network_student");
    _saveSwaNetworkFunction[NetworkType_Teacher] = LoadFunction(module, "save_swa_network_teacher");
    _saveSwaNetworkFunction[NetworkType_Student] = LoadFunction(module, "save_swa_network_student");
    _updateNetworkWeightsFunction = LoadFunction(module, "update_network_weights");
    _getNetworkInfoFunction[NetworkType_Teacher] = LoadFunction(module, "get_network_info_teacher");
    _getNetworkInfoFunction[NetworkType_Student] = LoadFunction(module, "get_network_info_student");
    _saveFileFunction = LoadFunction(module, "save_file");
    _loadFileFunction = LoadFunction(module, "load_file");
    _listChunksFunction = LoadFunction(module, "list_chunks");
    _fileExistsFunction = LoadFunction(module, "file_exists");
    _launchGuiFunction = LoadFunction(module, "launch_gui");
    _updateGuiFunction = LoadFunction(module, "update_gui");
    _debugDecompressFunction = LoadFunction(module, "debug_decompress");
    _optimizeParametersFunction = LoadFunction(module, "optimize_parameters");
    _runBotFunction = LoadFunction(module, "run_bot");
    _playBotMoveFunction = LoadFunction(module, "play_bot_move");

    Py_DECREF(module);
    Py_DECREF(pythonPath);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

PythonNetwork::~PythonNetwork()
{
    // Deallocating long-term Python objects like our API functions and the interpreter isn't really necessary
    // because we don't allocate multiple (e.g. even when training, the new strength test worker groups use the single network),
    // so there aren't any obvious possibilities for leaks/growth throughout the process, and the process exiting takes care of everything.
    //
    // However, explicitly deallocating can be useful when using memory leak detection tools, so it's often good practice
    // to do so. Our utilities may be lazy and explicitly finalize Python, but leave the PythonNetwork around until later
    // (e.g. on the class instead of in a work method), so check for Py_IsInitialized() to see if it's still safe to deallocate here.
    //
    // Note that relying on Py_FinalizeEx() to reclaim memory (e.g. for leak detection) has acknowledged bugs:
    // "Small amounts of memory allocated by the Python interpreter may not be freed (if you find a leak, please report it)."
    // "Memory tied up in circular references between objects is not freed."
    // "Some memory allocated by extension modules may not be freed."
    if (!Py_IsInitialized())
    {
        return;
    }

    Py_XDECREF(_playBotMoveFunction);
    Py_XDECREF(_runBotFunction);
    Py_XDECREF(_optimizeParametersFunction);
    Py_XDECREF(_debugDecompressFunction);
    Py_XDECREF(_fileExistsFunction);
    Py_XDECREF(_updateGuiFunction);
    Py_XDECREF(_launchGuiFunction);
    Py_XDECREF(_listChunksFunction);
    Py_XDECREF(_loadFileFunction);
    Py_XDECREF(_saveFileFunction);
    Py_XDECREF(_getNetworkInfoFunction[NetworkType_Student]);
    Py_XDECREF(_getNetworkInfoFunction[NetworkType_Teacher]);
    Py_XDECREF(_updateNetworkWeightsFunction);
    Py_XDECREF(_saveSwaNetworkFunction[NetworkType_Student]);
    Py_XDECREF(_saveSwaNetworkFunction[NetworkType_Teacher]);
    Py_XDECREF(_saveNetworkFunction[NetworkType_Student]);
    Py_XDECREF(_saveNetworkFunction[NetworkType_Teacher]);
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

std::vector<std::string> PythonNetwork::PredictCommentaryBatch(int batchSize, CommentaryInputPlanes* images)
{
    PythonContext context;

    // Make the predict call.
    npy_intp imageDims[2]{ batchSize, CommentaryInputPlaneCount };
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
        PyObject* pythonComment = *reinterpret_cast<PyObject**>(PyArray_GETPTR1(pythonCommentsArray, i));
        PyBytes_Check(pythonComment);

        const Py_ssize_t size = PyBytes_GET_SIZE(pythonComment);
        const char* data = PyBytes_AS_STRING(pythonComment);
        comments[i] = std::string(data, size);
    }

    Py_DECREF(result);
    Py_DECREF(pythonImages);

    return comments;
}

void PythonNetwork::Train(NetworkType networkType, int step, int checkpoint)
{
    PythonContext context;

    PyObject* pythonStep = PyLong_FromLong(step);
    PyAssert(pythonStep);

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyAssert(pythonCheckpoint);

    PyObject* result = PyObject_CallFunctionObjArgs(_trainFunction[networkType], pythonStep, pythonCheckpoint, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonCheckpoint);
    Py_DECREF(pythonStep);
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

void PythonNetwork::SaveSwaNetwork(NetworkType networkType, int checkpoint)
{
    PythonContext context;

    PyObject* pythonCheckpoint = PyLong_FromLong(checkpoint);
    PyAssert(pythonCheckpoint);

    PyObject* result = PyObject_CallFunctionObjArgs(_saveSwaNetworkFunction[networkType], pythonCheckpoint, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonCheckpoint);
}

void PythonNetwork::UpdateNetworkWeights(const std::string& networkWeights)
{
    PythonContext context;

    PyObject* pythonNetworkWeights = PyUnicode_FromStringAndSize(networkWeights.data(), networkWeights.size());
    PyAssert(pythonNetworkWeights);

    PyObject* result = PyObject_CallFunctionObjArgs(_updateNetworkWeightsFunction, pythonNetworkWeights, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonNetworkWeights);
}

void PythonNetwork::GetNetworkInfo(NetworkType networkType, int* stepCountOut, int* swaStepCountOut, int* trainingChunkCountOut, std::string* relativePathOut)
{
    PythonContext context;

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_getNetworkInfoFunction[networkType], nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract info.
    PyObject* pythonStepCount = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonStepCount);
    PyAssert(PyLong_Check(pythonStepCount));

    PyObject* pythonSwaStepCount = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonSwaStepCount);
    PyAssert(PyLong_Check(pythonSwaStepCount));

    PyObject* pythonTrainingChunkCount = PyTuple_GetItem(tupleResult, 2); // PyTuple_GetItem does not INCREF
    PyAssert(pythonTrainingChunkCount);
    PyAssert(PyLong_Check(pythonTrainingChunkCount));

    PyObject* pythonRelativePath = PyTuple_GetItem(tupleResult, 3); // PyTuple_GetItem does not INCREF
    PyAssert(pythonRelativePath);
    PyAssert(PyBytes_Check(pythonRelativePath));

    if (stepCountOut)
    {
        *stepCountOut = PyLong_AsLong(pythonStepCount);
    }
    if (swaStepCountOut)
    {
        *swaStepCountOut = PyLong_AsLong(pythonSwaStepCount);
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

std::vector<std::string> PythonNetwork::ListChunks()
{
    PythonContext context;

    PyObject* result = PyObject_CallFunctionObjArgs(_listChunksFunction, nullptr);
    PyAssert(result);
    PyAssert(PyList_Check(result));

    // Extract the filenames.
    std::vector<std::string> filenames;
    const int length = static_cast<int>(PyList_Size(result));
    for (int i = 0; i < length; i++)
    {
        PyObject* pythonFilename = PyList_GetItem(result, i);
        PyBytes_Check(pythonFilename);

        const Py_ssize_t size = PyBytes_GET_SIZE(pythonFilename);
        const char* data = PyBytes_AS_STRING(pythonFilename);
        filenames.emplace_back(data, size);
    }

    Py_DECREF(result);

    return filenames;
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

void PythonNetwork::UpdateGui(const std::string& fen, const std::string& line, int nodeCount, const std::string& evaluation, const std::string& principalVariation,
    const std::vector<std::string>& sans, const std::vector<std::string>& froms, const std::vector<std::string>& tos, std::vector<float>& targets,
    std::vector<float>& priors, std::vector<float>& values, std::vector<float>& puct, std::vector<int>& visits, std::vector<int>& weights)
{
    PythonContext context;

    PyObject* pythonFen = PyUnicode_FromStringAndSize(fen.data(), fen.size());
    PyAssert(pythonFen);

    PyObject* pythonLine = PyUnicode_FromStringAndSize(line.data(), line.size());
    PyAssert(pythonLine);

    PyObject* pythonNodeCount = PyLong_FromLong(nodeCount);
    PyAssert(pythonNodeCount);

    PyObject* pythonEvaluation = PyUnicode_FromStringAndSize(evaluation.data(), evaluation.size());
    PyAssert(pythonEvaluation);

    PyObject* pythonPrincipalVariation = PyUnicode_FromStringAndSize(principalVariation.data(), principalVariation.size());
    PyAssert(pythonPrincipalVariation);

    PyObject* pythonSans = PackNumpyStringArray(sans);
    PyObject* pythonFroms = PackNumpyStringArray(froms);
    PyObject* pythonTos = PackNumpyStringArray(tos);

    npy_intp moveDims[1]{ static_cast<int>(targets.size()) };
    PyObject* pythonTargets = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_FLOAT32, targets.data());
    PythonNetwork::PyAssert(pythonTargets);

    PyObject* pythonPriors = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_FLOAT32, priors.data());
    PythonNetwork::PyAssert(pythonPriors);

    PyObject* pythonValues = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_FLOAT32, values.data());
    PythonNetwork::PyAssert(pythonValues);

    PyObject* pythonPuct = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_FLOAT32, puct.data());
    PythonNetwork::PyAssert(pythonPuct);

    PyObject* pythonVisits = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_INT32, visits.data());
    PythonNetwork::PyAssert(pythonVisits);

    PyObject* pythonWeights = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_INT32, weights.data());
    PythonNetwork::PyAssert(pythonWeights);

    PyObject* result = PyObject_CallFunctionObjArgs(_updateGuiFunction, pythonFen, pythonLine, pythonNodeCount, pythonEvaluation, pythonPrincipalVariation,
        pythonSans, pythonFroms, pythonTos, pythonTargets, pythonPriors, pythonValues, pythonPuct, pythonVisits, pythonWeights, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonWeights);
    Py_DECREF(pythonVisits);
    Py_DECREF(pythonPuct);
    Py_DECREF(pythonValues);
    Py_DECREF(pythonPriors);
    Py_DECREF(pythonTargets);
    Py_DECREF(pythonTos);
    Py_DECREF(pythonFroms);
    Py_DECREF(pythonSans);
    Py_DECREF(pythonPrincipalVariation);
    Py_DECREF(pythonEvaluation);
    Py_DECREF(pythonNodeCount);
    Py_DECREF(pythonLine);
    Py_DECREF(pythonFen);
}

void PythonNetwork::DebugDecompress(int positionCount, int policySize, float* result, int64_t* imagePiecesAuxiliary,
    int64_t* policyRowLengths, int64_t* policyIndices, float* policyValues, int decompressPositionsModulus,
    InputPlanes* imagesOut, float* valuesOut, OutputPlanes* policiesOut)
{
    PythonContext context;

    // Compressed probabilities are already in Python/TensorFlow [-1, 1] range (see Storage::PopulateGame).

    // Wrap compressed data in numpy.
    PyObject* pythonResult = PyArray_SimpleNewFromData(0, 0, NPY_FLOAT32, result);
    PyAssert(pythonResult);

    npy_intp imagePiecesAuxiliaryDims[2]{ positionCount, InputPieceAndRepetitionPlanesPerPosition + InputAuxiliaryPlaneCount };
    PyObject* pythonImagePiecesAuxiliary = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imagePiecesAuxiliaryDims), imagePiecesAuxiliaryDims, NPY_INT64, imagePiecesAuxiliary);
    PyAssert(pythonImagePiecesAuxiliary);

    npy_intp perPositionDims[1]{ positionCount };
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

    PyObject* pythonDecompressPositionsModulus = PyLong_FromLong(decompressPositionsModulus);
    PyAssert(pythonDecompressPositionsModulus);

    // Make the call.
    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_debugDecompressFunction, pythonResult, pythonImagePiecesAuxiliary,
        pythonPolicyRowLengths, pythonPolicyIndices, pythonPolicyValues, pythonDecompressPositionsModulus, nullptr);
    PyAssert(tupleResult);
    PyAssert(PyTuple_Check(tupleResult));

    // Extract the images.
    PyObject* pythonImages = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    PyAssert(pythonImages);
    PyAssert(PyArray_Check(pythonImages));

    PyArrayObject* pythonImagesArray = reinterpret_cast<PyArrayObject*>(pythonImages);
    PackedPlane* pythonImagesPtr = reinterpret_cast<PackedPlane*>(PyArray_DATA(pythonImagesArray));

    const int outputPositionCount = (positionCount / decompressPositionsModulus);
    const int imageCount = (outputPositionCount * InputPlaneCount);
    std::copy(pythonImagesPtr, pythonImagesPtr + imageCount, reinterpret_cast<PackedPlane*>(imagesOut));

    // Extract the values.
    PyObject* pythonValues = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    PyAssert(pythonValues);
    PyAssert(PyArray_Check(pythonValues));

    PyArrayObject* pythonValuesArray = reinterpret_cast<PyArrayObject*>(pythonValues);
    float* pythonValuesPtr = reinterpret_cast<float*>(PyArray_DATA(pythonValuesArray));

    const int valueCount = outputPositionCount;
    std::copy(pythonValuesPtr, pythonValuesPtr + valueCount, valuesOut);

    // Network deals with tanh outputs/targets in (-1, 1)/[-1, 1]. MCTS deals with probabilities in [0, 1].
    MapProbabilities11To01(valueCount, valuesOut);

    // Extract the policies.
    PyObject* pythonPolicies = PyTuple_GetItem(tupleResult, 2); // PyTuple_GetItem does not INCREF
    PyAssert(pythonPolicies);
    PyAssert(PyArray_Check(pythonPolicies));

    PyArrayObject* pythonPoliciesArray = reinterpret_cast<PyArrayObject*>(pythonPolicies);
    PlanesPointerFlat pythonPoliciesPtr = reinterpret_cast<PlanesPointerFlat>(PyArray_DATA(pythonPoliciesArray));

    const int policyCount = (outputPositionCount * OutputPlanesFloatCount);
    std::copy(pythonPoliciesPtr, pythonPoliciesPtr + policyCount, reinterpret_cast<PlanesPointerFlat>(policiesOut));

    Py_DECREF(tupleResult);
    Py_DECREF(pythonDecompressPositionsModulus);
    Py_DECREF(pythonPolicyValues);
    Py_DECREF(pythonPolicyIndices);
    Py_DECREF(pythonPolicyRowLengths);
    Py_DECREF(pythonImagePiecesAuxiliary);
    Py_DECREF(pythonResult);
}

void PythonNetwork::OptimizeParameters()
{
    PythonContext context;

    PyObject* result = PyObject_CallFunctionObjArgs(_optimizeParametersFunction, nullptr);
    PyAssert(result);

    Py_DECREF(result);
}

void PythonNetwork::RunBot()
{
    PythonContext context;

    PyObject* result = PyObject_CallFunctionObjArgs(_runBotFunction, nullptr);
    PyAssert(result);

    Py_DECREF(result);
}

void PythonNetwork::PlayBotMove(const std::string& gameId, const std::string& move)
{
    PythonContext context;

    PyObject* pythonGameId = PyUnicode_FromStringAndSize(gameId.data(), gameId.size());
    PyAssert(pythonGameId);

    PyObject* pythonMove = PyUnicode_FromStringAndSize(move.data(), move.size());
    PyAssert(pythonMove);

    PyObject* result = PyObject_CallFunctionObjArgs(_playBotMoveFunction, pythonGameId, pythonMove, nullptr);
    PyAssert(result);

    Py_DECREF(result);
    Py_DECREF(pythonMove);
    Py_DECREF(pythonGameId);
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