#include "PythonNetwork.h"

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

RawPrediction::RawPrediction(float value, OutputPlanesPtr policy)
    : _value(value)
{
    constexpr int count = sizeof(_policy) / sizeof(float);
    std::copy(reinterpret_cast<float*>(policy), reinterpret_cast<float*>(policy) + count, reinterpret_cast<float*>(_policy.data()));
}

float RawPrediction::Value() const
{
    return _value;
}

void* RawPrediction::Policy()
{
    return reinterpret_cast<void*>(_policy.data());
}

BatchedPythonNetwork::BatchedPythonNetwork()
{
    PythonContext context;

    _module = PyImport_ImportModule("network");
    assert(_module);

    _predictBatchFunction = PyObject_GetAttrString(_module, "predict_batch");
    assert(_predictBatchFunction);
    assert(PyCallable_Check(_predictBatchFunction));

    _submitFunction = PyObject_GetAttrString(_module, "submit");
    assert(_submitFunction);
    assert(PyCallable_Check(_submitFunction));
}

BatchedPythonNetwork::~BatchedPythonNetwork()
{
    Py_XDECREF(_submitFunction);
    Py_XDECREF(_predictBatchFunction);
    Py_XDECREF(_module);
}

IPrediction* BatchedPythonNetwork::Predict(InputPlanes& image)
{
    SyncQueue<IPrediction*> output;
    
    // Safely push to the worker queue.
    {
        std::unique_lock lock(_mutex);

        _predictQueue.emplace_back(&image, &output);

        // Only send a wake-up on the (BatchSize-1) to BatchSize transition, to avoid spam.
        if (_predictQueue.size() == BatchSize)
        {
            _condition.notify_one();
        }
    }

    // Wait for the worker thread to process the batch.
    return output.Pop();
}

void BatchedPythonNetwork::Submit(float terminalValue,
    std::vector<int>& moves,
    std::vector<InputPlanes>& images,
    std::vector<OutputPlanes>& policies)
{
    // TODO: Don't queue/batch/etc. for now while we have one game thread.
    
    PythonContext context;

    PyObject* pythonTerminalValue = PyFloat_FromDouble(terminalValue);
    assert(pythonTerminalValue);

    npy_intp moveDims[1]{ static_cast<int>(moves.size()) };
    PyObject* pythonMoves = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(moveDims), moveDims, NPY_INT32, reinterpret_cast<void*>(moves.data()));
    assert(pythonMoves);

    npy_intp imageDims[4]{ static_cast<int>(images.size()), 12, 8, 8 };
    PyObject* pythonImages = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(imageDims), imageDims, NPY_FLOAT32, reinterpret_cast<void*>(images.data()));
    assert(pythonImages);

    npy_intp policyDims[4]{ static_cast<int>(policies.size()), 73, 8, 8 };
    PyObject* pythonPolicies = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(policyDims), policyDims, NPY_FLOAT32, reinterpret_cast<void*>(policies.data()));
    assert(pythonPolicies);

    PyObject* result = PyObject_CallFunctionObjArgs(_submitFunction, pythonTerminalValue, pythonMoves, pythonImages, pythonPolicies, nullptr);
    assert(result);

    Py_DECREF(result);
    Py_DECREF(pythonPolicies);
    Py_DECREF(pythonImages);
    Py_DECREF(pythonMoves);
    Py_DECREF(pythonTerminalValue);
}

void BatchedPythonNetwork::Work()
{
    while (true)
    {
        // Wait until the queue is full enough to process.
        std::unique_lock lock(_mutex);
        _condition.wait(lock, [&] { return (_predictQueue.size() >= BatchSize); });

        // Drain and unlock as quickly as possible.
        std::vector<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>> batch(BatchSize);
        for (int i = 0; i < BatchSize; i++)
        {
            batch[i] = _predictQueue.front();
            _predictQueue.pop_front();
        }
        lock.unlock();

        // Combine images.
        npy_intp dims[4]{ BatchSize, 12, 8, 8 };
        std::unique_ptr<std::array<std::array<std::array<std::array<float, 8>, 8>, 12>, BatchSize>> batchImage(
            new std::array<std::array<std::array<std::array<float, 8>, 8>, 12>, BatchSize>());
        for (int i = 0; i < BatchSize; i++)
        {
            (*batchImage)[i] = *batch[i].first;
        }

        {
            PythonContext context;

            // Make the predict call.
            PyObject* pythonBatchImage = PyArray_SimpleNewFromData(
                Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT32, reinterpret_cast<void*>(batchImage->data()));
            assert(pythonBatchImage);

            PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictBatchFunction, pythonBatchImage, nullptr);
            assert(tupleResult);

            // Extract the values.
            PyObject* pythonValue = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
            assert(PyArray_Check(pythonValue));

            PyArrayObject* pythonValueArray = reinterpret_cast<PyArrayObject*>(pythonValue);
            float* values = reinterpret_cast<float*>(PyArray_DATA(pythonValueArray));

            // Extract the policies.
            PyObject* pythonPolicy = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
            assert(PyArray_Check(pythonPolicy));

            PyArrayObject* pythonPolicyArray = reinterpret_cast<PyArrayObject*>(pythonPolicy);
            float(*policies)[73][8][8] = reinterpret_cast<float(*)[73][8][8]>(PyArray_DATA(pythonPolicyArray));

            // Deliver predictions.
            for (int i = 0; i < BatchSize; i++)
            {
                batch[i].second->Push(new RawPrediction(values[i], policies[i]));
            }

            Py_DECREF(tupleResult);
            Py_DECREF(pythonBatchImage);
        }
    }
}

void BatchedPythonNetwork::Train()
{
    PythonContext context;

    PyObject* trainFunction = PyObject_GetAttrString(_module, "train");
    assert(trainFunction);
    assert(PyCallable_Check(trainFunction));

    // Make the call and train forever.
    PyObject* result = PyObject_CallFunctionObjArgs(trainFunction, nullptr);
    assert(result);
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

void* UniformPrediction::Policy()
{
    return _policy;
}

UniformNetwork::UniformNetwork()
{
}

UniformNetwork::~UniformNetwork()
{
}

IPrediction* UniformNetwork::Predict(InputPlanes& image)
{
    return new UniformPrediction(reinterpret_cast<void*>(_policy.data()));
}

void UniformNetwork::Submit(float terminalValue,
    std::vector<int>& moves,
    std::vector<InputPlanes>& images,
    std::vector<OutputPlanes>& policies)
{
    _batchedPythonNetwork.Submit(terminalValue, moves, images, policies);
}

void UniformNetwork::Train()
{
    _batchedPythonNetwork.Train();
}