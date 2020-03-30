#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>

#include "Config.h"

typedef std::array<std::array<std::array<float, 8>, 8>, 12> InputPlanes;
typedef std::array<std::array<std::array<float, 8>, 8>, 73> OutputPlanes;

struct INetwork
{
    static const int PredictionBatchSize = 64;

    virtual ~INetwork() {};

    virtual void PredictBatch(InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void TrainBatch(int step, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void SaveNetwork(int checkpoint) = 0;
};

#endif // _NETWORK_H_