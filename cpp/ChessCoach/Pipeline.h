#ifndef _PIPELINE_H_
#define _PIPELINE_H_

#include <vector>
#include <mutex>

#include "Network.h"
#include "ReplayBuffer.h"

class Pipeline
{
private:

    // Don't completely fill the buffer, to avoid clobbbering the in-use sampled batch.
    static const int BufferCount = 16;
    static const int MaxFill = (BufferCount - 1);

public:

    void Initialize(const ReplayBuffer& games, int trainingBatchSize, int workerCount);
    TrainingBatch* SampleBatch();

private:

    void GenerateBatches();
    void AddBatch(TrainingBatch&& batch);

private:

    const ReplayBuffer* _games = nullptr;
    int _trainingBatchSize = 1;
    std::array<TrainingBatch, BufferCount> _batches;
    std::mutex _mutex;
    std::condition_variable _batchExists;
    std::condition_variable _roomExists;

    int _oldest = 0;
    int _count = 0;
};

#endif // _PIPELINE_H_