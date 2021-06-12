#ifndef _WORKERGROUP_H_
#define _WORKERGROUP_H_

#include <vector>
#include <thread>

#include "SelfPlay.h"

class WorkerGroup
{
public:

    bool IsInitialized();
    void ShutDown();

    template <typename Function>
    void Initialize(INetwork* network, Storage* storage, NetworkType networkType, int workerCount, int workerParallelism, Function workerLoop)
    {
        workCoordinator.reset(new WorkCoordinator(workerCount));
        controllerWorker.reset(new SelfPlayWorker(storage, &searchState, 1 /* gameCount */));
        controllerWorker->Initialize();
        for (int i = 0; i < workerCount; i++)
        {
            selfPlayWorkers.emplace_back(new SelfPlayWorker(storage, &searchState, workerParallelism));
            selfPlayThreads.emplace_back(workerLoop, selfPlayWorkers[i].get(), workCoordinator.get(), network, networkType, i);
        }
    }

    SearchState searchState{};
    std::unique_ptr<WorkCoordinator> workCoordinator;
    std::unique_ptr<SelfPlayWorker> controllerWorker;
    std::vector<std::unique_ptr<SelfPlayWorker>> selfPlayWorkers;
    std::vector<std::thread> selfPlayThreads;
};

#endif // _WORKERGROUP_H_