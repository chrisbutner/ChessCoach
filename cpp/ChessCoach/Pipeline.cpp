#include "Pipeline.h"

#include "Random.h"

using namespace std::chrono_literals;

Pipeline::~Pipeline()
{
    {
        std::unique_lock lock(_mutex);

        _shuttingDown = true;

        _roomExists.notify_all();
    }

    for (std::thread& thread : _threads)
    {
        thread.join();
    }
}

void Pipeline::Initialize(const ReplayBuffer& games, int trainingBatchSize, int workerCount)
{
    if (_initialized)
    {
        throw std::runtime_error("Pipeline already initialized");
    }
    _initialized = true;

    _games = &games;
    _trainingBatchSize = trainingBatchSize;

    for (int i = 0; i < workerCount; i++)
    {
        _threads.emplace_back(&Pipeline::GenerateBatches, this);
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

    while (!_shuttingDown && (_count >= MaxFill))
    {
        _roomExists.wait(lock);
    }
    if (_shuttingDown)
    {
        return;
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
    workingBatch.mctsValues.resize(_trainingBatchSize);
    workingBatch.policies.resize(_trainingBatchSize);
    workingBatch.replyPolicies.resize(_trainingBatchSize);

    while (!_shuttingDown)
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
        workingBatch.mctsValues.resize(_trainingBatchSize);
        workingBatch.policies.resize(_trainingBatchSize);
        workingBatch.replyPolicies.resize(_trainingBatchSize);
    }
}