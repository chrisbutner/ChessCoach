#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>

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

    virtual IPrediction* Predict(InputPlanes& image) = 0;
};

#endif // _NETWORK_H_