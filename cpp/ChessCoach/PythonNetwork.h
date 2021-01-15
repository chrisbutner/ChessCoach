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

#define PY_ARRAY_UNIQUE_SYMBOL ChessCoach_ArrayApi
#define NO_IMPORT_ARRAY

class PythonContext
{
private:

    thread_local static PyGILState_STATE GilState;
    thread_local static PyThreadState* ThreadState;

public:

    PythonContext();
    ~PythonContext();
};

class NonPythonContext
{
public:

    NonPythonContext();
    ~NonPythonContext();

private:

    PyThreadState* _threadState;
};

class PythonNetwork : public INetwork
{
public:

    static void PyAssert(bool result);
    static PyObject* PackNumpyStringArray(const std::vector<std::string>& values);

public:

    PythonNetwork();
    virtual ~PythonNetwork();

    virtual PredictionStatus PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies);
    virtual std::vector<std::string> PredictCommentaryBatch(int batchSize, InputPlanes* images);
    virtual void Train(NetworkType networkType, std::vector<GameType>& gameTypes,
        std::vector<Window>& trainingWindows, int step, int checkpoint);
    virtual void TrainCommentary(int step, int checkpoint);
    virtual void LogScalars(NetworkType networkType, int step, const std::vector<std::string> names, float* values);
    virtual void LoadNetwork(const std::string& networkName);
    virtual void SaveNetwork(NetworkType networkType, int checkpoint);
    virtual void GetNetworkInfo(NetworkType networkType, int* stepCountOut, int* trainingChunkCountOut, std::string* relativePathOut);
    virtual void SaveFile(const std::string& relativePath, const std::string& data);
    virtual std::string LoadFile(const std::string& relativePath);
    virtual bool FileExists(const std::string& relativePath);
    virtual void LaunchGui(const std::string& mode);
    virtual void UpdateGui(const std::string& fen, int nodeCount, const std::string& evaluation, const std::string& principleVariation,
        const std::vector<std::string>& sans, const std::vector<std::string>& froms, const std::vector<std::string>& tos, std::vector<float>& targets,
        std::vector<float>& priors, std::vector<float>& values, std::vector<float>& ucb, std::vector<int>& visits, std::vector<int>& weights);
    virtual void DebugDecompress(int positionCount, int policySize, float* result, int64_t* imagePiecesAuxiliary,
        int64_t* policyRowLengths, int64_t* policyIndices, float* policyValues, int decompressPositionsModulus,
        InputPlanes* imagesOut, float* valuesOut, OutputPlanes* policiesOut);
    virtual void OptimizeParameters();

private:

    PyObject* LoadFunction(PyObject* module, const char* name);

private:

    PyObject* _predictBatchFunction[NetworkType_Count];
    PyObject* _predictCommentaryBatchFunction;
    PyObject* _trainFunction[NetworkType_Count];
    PyObject* _trainCommentaryFunction;
    PyObject* _logScalarsFunction[NetworkType_Count];
    PyObject* _loadNetworkFunction;
    PyObject* _saveNetworkFunction[NetworkType_Count];
    PyObject* _getNetworkInfoFunction[NetworkType_Count];
    PyObject* _saveFileFunction;
    PyObject* _loadFileFunction;
    PyObject* _fileExistsFunction;
    PyObject* _launchGuiFunction;
    PyObject* _updateGuiFunction;
    PyObject* _debugDecompressFunction;
    PyObject* _optimizeParametersFunction;
};

#endif // _PYTHONNETWORK_H_