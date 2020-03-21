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

BatchedPythonPrediction::BatchedPythonPrediction(PyObject* tupleResult, int index)
    : _tupleResult(tupleResult)
{
    assert(tupleResult);

    // Increase the refcount so that the result tuple isn't destroyed until all predictions are.
    if (index > 0)
    {
        Py_INCREF(tupleResult);
    }

    // Extract the value.
    PyObject* pythonValue = PyTuple_GetItem(tupleResult, 0); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonValue));

    PyArrayObject* pythonValueArray = reinterpret_cast<PyArrayObject*>(pythonValue);
    _value = reinterpret_cast<float*>(PyArray_DATA(pythonValueArray))[index];

    // Extract the policy.
    PyObject* pythonPolicy = PyTuple_GetItem(tupleResult, 1); // PyTuple_GetItem does not INCREF
    assert(PyArray_Check(pythonPolicy));

    PyArrayObject* pythonPolicyArray = reinterpret_cast<PyArrayObject*>(pythonPolicy);
    _policy = reinterpret_cast<void*>(
        reinterpret_cast<float(*)[BatchedPythonNetwork::BatchSize][73][8][8]>(
            PyArray_DATA(pythonPolicyArray))
        [index]);
}

BatchedPythonPrediction::~BatchedPythonPrediction()
{
    Py_XDECREF(_tupleResult);
}

float BatchedPythonPrediction::Value() const
{
    return _value;
}

void* BatchedPythonPrediction::Policy() const
{
    return _policy;
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
    std::unique_lock lock(_mutex);

    // The SyncQueue needs to out-live the queue pointer.
    SyncQueue<IPrediction*> output;
    _predictQueue.emplace_back(&image, &output);

    if (_predictQueue.size() >= BatchSize)
    {
        // The queue is full enough to process, so this predictor does the work.
        assert(_predictQueue.size() == BatchSize);

        // Steal the queue's storage to unlock as quickly as possible and let other predictors in for the next batch
        // while delivering predictions back to this batch.
        std::vector<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>> predictQueue(std::move(_predictQueue));
        
        // Go with "radioactive" and reinitialize rather than clear.
        // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3241.html
        _predictQueue = std::vector<std::pair<InputPlanes*, SyncQueue<IPrediction*>*>>();
        
        // Combine images.
        npy_intp dims[4]{ BatchSize, 12, 8, 8 };
        std::unique_ptr<std::array<std::array<std::array<std::array<float, 8>, 8>, 12>, BatchSize>> batchImage(
            new std::array<std::array<std::array<std::array<float, 8>, 8>, 12>, BatchSize>());
        for (int i = 0; i < BatchSize; i++)
        {
            (*batchImage)[i] = *predictQueue[i].first;
        }

        PyObject* pythonBatchImage = PyArray_SimpleNewFromData(
            Py_ARRAY_LENGTH(dims), dims, NPY_FLOAT, reinterpret_cast<void*>(batchImage->data()));
        assert(pythonBatchImage);

        PyObject* tupleResult = PyObject_CallFunctionObjArgs(_predictBatchFunction, pythonBatchImage, nullptr);
        assert(tupleResult);

        Py_DECREF(pythonBatchImage);

        // We've finished calling into Python, so it's safe to let the next batch in.
        lock.unlock();

        // Deliver predictions, including to self.
        for (int i = 0; i < BatchSize; i++)
        {
            predictQueue[i].second->Push(new BatchedPythonPrediction(tupleResult, i));
        }

        return output.Pop();
    }
    else
    {
        // Not ready, so unlock and wait for another predictor to do the work.
        lock.unlock();
        return output.Pop();
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

IPrediction* UniformNetwork::Predict(InputPlanes& image)
{
    return new UniformPrediction(reinterpret_cast<void*>(_policy.data()));
}