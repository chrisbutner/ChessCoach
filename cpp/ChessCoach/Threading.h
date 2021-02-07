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

class WorkCoordinator
{
public:

    WorkCoordinator(int workerCount);

    void OnWorkItemCompleted();
    bool AllWorkItemsCompleted();

    void ResetWorkItemsRemaining(int workItemsRemaining);
    void ShutDown();

    bool WaitForWorkItems();
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
    bool _shutDown;

    int _workerCount;
    int _workerReadyCount;

    // Rely on fencing via _mutex.
    bool _generateUniformPredictions;
};

#endif // _THREADING_H_