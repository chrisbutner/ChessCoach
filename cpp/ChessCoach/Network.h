#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>

#include "Config.h"

struct INetwork
{
    static const int BoardSide = 8;
    static const int InputPlaneCount = 25;
    static const int OutputPlaneCount = 73;

    static const int PlaneFloatCount = BoardSide * BoardSide;
    static const int InputPlanesFloatCount = PlaneFloatCount * InputPlaneCount;
    static const int OutputPlanesFloatCount = PlaneFloatCount * OutputPlaneCount;

    typedef std::array <std::array<float, BoardSide>, BoardSide> Plane;
    typedef std::array<Plane, InputPlaneCount> InputPlanes;
    typedef std::array<Plane, OutputPlaneCount> OutputPlanes;

    virtual ~INetwork() {};

    virtual void PredictBatch(InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void TrainBatch(int step, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void SaveNetwork(int checkpoint) = 0;
};

#endif // _NETWORK_H_