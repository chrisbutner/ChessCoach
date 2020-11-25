#ifndef _THREADING_H_
#define _THREADING_H_

#include <cassert>
#include <mutex>
#include <condition_variable>
#include <atomic>

class Throttle
{
public:

    Throttle(int durationMilliseconds);

    bool TryFire();

private:

    int _durationMilliseconds;
    std::atomic_int64_t _last;
};

template <typename T>
class SyncQueue
{
public:

    SyncQueue()
        : _hasItem(false)
        , _item()
    {
    }

    void Push(const T& item)
    {
        std::lock_guard lock(_mutex);
        assert(!_hasItem);
        _hasItem = true;
        _item = item;
        _condition.notify_one();
    }

    T Pop()
    {
        std::unique_lock lock(_mutex);
        _condition.wait(lock, [&]{ return _hasItem; });
        _hasItem = false;
        return _item;
    }

    bool Poll(T& item)
    {
        std::lock_guard lock(_mutex);
        if (_hasItem)
        {
            _hasItem = false;
            item = _item;
            return true;
        }
        return false;
    }

private:

    std::mutex _mutex;
    std::condition_variable _condition;
    bool _hasItem;
    T _item;
};

class WorkCoordinator
{
public:

    WorkCoordinator(int workerCount);

    void OnWorkItemCompleted();
    bool AllWorkItemsCompleted();

    void ResetWorkItemsRemaining(int workItemsRemaining);

    void WaitForWorkItems();
    void WaitForWorkers();
    bool WaitForWorkers(int timeoutMilliseconds);

    bool CheckWorkItemsExist();
    bool CheckWorkersReady();

    bool& GenerateUniformPredictions();

private:

    std::mutex _mutex;
    std::condition_variable _workItemsExist;
    std::condition_variable _workersReady;

    // Atomic is not needed for the locks/waits, but is for OnWorkItemCompleted/AllWorkItemsCompleted.
    std::atomic_int _workItemsRemaining;

    int _workerCount;
    int _workerReadyCount;

    // Rely on fencing via _mutex.
    bool _generateUniformPredictions;
};

#endif // _THREADING_H_