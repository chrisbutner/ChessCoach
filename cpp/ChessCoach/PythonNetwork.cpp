#include "PythonNetwork.h"

#include <vector>

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

void* PythonPrediction::Policy()
{
    return _policy;
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

IPrediction* PythonNetwork::Predict(InputPlanes& image)
{
    npy_intp dims[3]{ 12, 8, 8 };

    PyObject* pythonImage = PyArray_SimpleNewFromData(
        Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT, reinterpret_cast<void*>(image.data()));
    assert(pythonImage);

    PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictFunction, pythonImage, nullptr);
    assert(tupleResult);

    Py_DECREF(pythonImage);

    return new PythonPrediction(tupleResult);
}

BatchedPythonNetwork::BatchedPythonNetwork()
{
    if (!PyArray_API)
    {
        _import_array();
    }

    _predictModule = PyImport_ImportModule("predict");
    assert(_predictModule);

    _predictBatchFunction = PyObject_GetAttrString(_predictModule, "predict_batch");
    assert(_predictBatchFunction);
    assert(PyCallable_Check(_predictBatchFunction));
}

BatchedPythonNetwork::~BatchedPythonNetwork()
{
    Py_XDECREF(_predictBatchFunction);
    Py_XDECREF(_predictModule);
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

        PyObject* pythonBatchImage = PyArray_SimpleNewFromData(
            Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT, reinterpret_cast<void*>(batchImage->data()));
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