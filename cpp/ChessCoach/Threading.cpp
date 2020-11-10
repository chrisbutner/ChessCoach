#include "Threading.h"

#include <chrono>
#include <functional>

WorkCoordinator::WorkCoordinator(int workerCount)
    : _workItemsRemaining(0)
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

void WorkCoordinator::WaitForWorkItems()
{
    std::unique_lock lock(_mutex);

    _workerReadyCount++;

    if (CheckWorkersReady())
    {
        _workersReady.notify_all();
    }

    while (!CheckWorkItemsExist())
    {
        _workItemsExist.wait(lock);
    }

    _workerReadyCount--;
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