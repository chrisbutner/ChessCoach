// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

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