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