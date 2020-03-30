#include "Threading.h"

WorkCoordinator::WorkCoordinator(int workItemCount)
    : _workItemsRemaining(workItemCount)
{
}

void WorkCoordinator::OnWorkItemCompleted()
{
    --_workItemsRemaining;
}

bool WorkCoordinator::AllWorkItemsCompleted()
{
    return (_workItemsRemaining <= 0);
}