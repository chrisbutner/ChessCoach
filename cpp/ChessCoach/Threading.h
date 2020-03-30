#ifndef _THREADING_H_
#define _THREADING_H_

#include <cassert>
#include <mutex>
#include <condition_variable>
#include <atomic>

template <class T>
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

    WorkCoordinator(int workItemCount);

    void OnWorkItemCompleted();
    bool AllWorkItemsCompleted();

private:

    std::atomic_int _workItemsRemaining;
};

#endif // _THREADING_H_