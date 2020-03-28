#include "Threading.h"

WorkCoordinator::WorkCoordinator(int workItemCount)
    : _workItemsRemaining(workItemCount)
{
}

void WorkCoordinator::OnWorkItemCompleted()
{
    std::lock_guard lock(_mutex);

    if (--_workItemsRemaining <= 0)
    {
        _condition.notify_all();
    }
}

void WorkCoordinator::Wait()
{
    std::unique_lock lock(_mutex);
    _condition.wait(lock, [&] { return (_workItemsRemaining <= 0); });
}