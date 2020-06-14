#include "Pipeline.h"

#include "Random.h"

using namespace std::chrono_literals;

void Pipeline::Initialize(const ReplayBuffer& games, int trainingBatchSize, int workerCount)
{
    _games = &games;
    _trainingBatchSize = trainingBatchSize;

    for (int i = 0; i < workerCount; i++)
    {
        // Just leak them until clean ending/joining for the whole training process is implemented.
        new std::thread(&Pipeline::GenerateBatches, this);
    }
}

TrainingBatch* Pipeline::SampleBatch()
{
    std::unique_lock lock(_mutex);

    while (_count <= 0)
    {
        //std::cout << "SampleBatch pipeline starved" << std::endl;
        _batchExists.wait(lock);
    }

    const int index = ((_oldest + BufferCount - _count) % BufferCount);
    if (--_count == (MaxFill - 1))
    {
        _roomExists.notify_one();
    }
    return &_batches[index];
}

void Pipeline::AddBatch(TrainingBatch&& batch)
{
    std::unique_lock lock(_mutex);

    while (_count >= MaxFill)
    {
        _roomExists.wait(lock);
    }

    _batches[_oldest] = std::move(batch);

    _oldest = ((_oldest + 1) % BufferCount);
    if (++_count == 1)
    {
        _batchExists.notify_one();
    }
}

void Pipeline::GenerateBatches()
{
    TrainingBatch workingBatch;
    workingBatch.images.resize(_trainingBatchSize);
    workingBatch.values.resize(_trainingBatchSize);
    workingBatch.policies.resize(_trainingBatchSize);
    workingBatch.replyPolicies.resize(_trainingBatchSize);

    while (true)
    {
        if (!_games->SampleBatch(workingBatch))
        {
            // Not enough games yet.
            std::this_thread::sleep_for(5s);
            continue;
        }

        AddBatch(std::move(workingBatch));
        workingBatch.images.resize(_trainingBatchSize);
        workingBatch.values.resize(_trainingBatchSize);
        workingBatch.policies.resize(_trainingBatchSize);
        workingBatch.replyPolicies.resize(_trainingBatchSize);
    }
}