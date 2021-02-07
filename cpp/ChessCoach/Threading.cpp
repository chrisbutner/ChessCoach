#include "Threading.h"

#include <chrono>
#include <functional>

Throttle::Throttle(int durationMilliseconds)
    : _durationMilliseconds(durationMilliseconds)
    , _last(0)
{
}

bool Throttle::TryFire()
{
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    int64_t last = _last;

    while ((now - last) >= _durationMilliseconds)
    {
        if (_last.compare_exchange_weak(last, now))
        {
            return true;
        }
    }
    return false;
}

WorkCoordinator::WorkCoordinator(int workerCount)
    : _workItemsRemaining(0)
    , _shutDown(false)
    , _workerCount(workerCount)
    , _workerReadyCount(0)
{
}

void WorkCoordinator::OnWorkItemCompleted()
{
    // This may go below zero, which is fine for our use.
    --_workItemsRemaining;
}

bool WorkCoordinator::AllWorkItemsCompleted()
{
    return (_workItemsRemaining <= 0);
}

void WorkCoordinator::ResetWorkItemsRemaining(int workItemsRemaining)
{
    std::lock_guard lock(_mutex);

    _workItemsRemaining = workItemsRemaining;

    if (CheckWorkItemsExist())
    {
        _workItemsExist.notify_all();
    }
}

void WorkCoordinator::ShutDown()
{
    std::lock_guard lock(_mutex);

    _shutDown = true;
    _workItemsExist.notify_all();
}

// Returns true if work items found, false to shut down.
bool WorkCoordinator::WaitForWorkItems()
{
    std::unique_lock lock(_mutex);

    _workerReadyCount++;

    if (CheckWorkersReady())
    {
        _workersReady.notify_all();
    }

    while (!CheckWorkItemsExist() && !_shutDown)
    {
        _workItemsExist.wait(lock);
    }

    _workerReadyCount--;

    return !_shutDown;
}

void WorkCoordinator::WaitForWorkers()
{
    std::unique_lock lock(_mutex);

    while (!CheckWorkersReady())
    {
        _workersReady.wait(lock);
    }
}

bool WorkCoordinator::WaitForWorkers(int timeoutMilliseconds)
{
    std::unique_lock lock(_mutex);

    return _workersReady.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
        std::bind(&WorkCoordinator::CheckWorkersReady, this));
}

bool WorkCoordinator::CheckWorkItemsExist()
{
    return (_workItemsRemaining > 0);
}

bool WorkCoordinator::CheckWorkersReady()
{
    // Workers are only ready *for new work*. Otherwise they're just lazy
    // and haven't started yet.
    return !CheckWorkItemsExist() && (_workerReadyCount >= _workerCount);
}

bool& WorkCoordinator::GenerateUniformPredictions()
{
    return _generateUniformPredictions;
}