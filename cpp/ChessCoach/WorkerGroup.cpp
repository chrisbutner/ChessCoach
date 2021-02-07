#include "WorkerGroup.h"

bool WorkerGroup::IsInitialized()
{
    return workCoordinator.get();
}

void WorkerGroup::ShutDown()
{
    workCoordinator->ShutDown();
    for (std::thread& thread : selfPlayThreads)
    {
        thread.join();
    }
}