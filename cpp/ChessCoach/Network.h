#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>

#include "Config.h"

typedef std::array<std::array<std::array<float, 8>, 8>, 12> InputPlanes;
typedef std::array<std::array<std::array<float, 8>, 8>, 73> OutputPlanes;
typedef float(*OutputPlanesPtr)[8][8];

struct IPrediction
{
    virtual ~IPrediction() {};

    virtual float Value() const = 0;
    virtual void* Policy() = 0; // To view as an OutputPlanesPtr
};

struct INetwork
{
    virtual ~INetwork() {};

    virtual void SetEnabled(bool enabled) = 0;
    virtual IPrediction* Predict(InputPlanes& image) = 0;
    virtual void TrainBatch(int step, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void SaveNetwork(int checkpoint) = 0;
};

#endif // _NETWORK_H_