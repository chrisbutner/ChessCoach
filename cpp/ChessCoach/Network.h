#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>
#include <cassert>

#include "Config.h"

constexpr static const float NETWORK_VALUE_WIN = 1.f;
constexpr static const float NETWORK_VALUE_DRAW = 0.f;
constexpr static const float NETWORK_VALUE_LOSS = -1.f;

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

    constexpr static uint8_t QuantizeProbability(float probability01)
    {
        return static_cast<uint8_t>(probability01 * 255.f + 0.5f);
    }

    constexpr static float DequantizeProbability(uint8_t quantizedProbability)
    {
        return (quantizedProbability / 255.f);
    }

    constexpr static float MapProbability01To11(float probability01)
    {
        return ((probability01 * 2.f) - 1.f);
    }

    constexpr static float MapProbability11To01(float probability11)
    {
        return ((probability11 + 1.f) / 2.f);
    }

    inline static void MapProbabilities01To11(size_t count, float* probabilities)
    {
        for (int i = 0; i < count; i++)
        {
            // Check the more restrictive range before mapping.
            assert((probabilities[i] >= 0.f) && (probabilities[i] <= 1.f));

            probabilities[i] = MapProbability01To11(probabilities[i]);
        }
    }

    inline static void MapProbabilities11To01(size_t count, float* probabilities)
    {
        for (int i = 0; i < count; i++)
        {
            probabilities[i] = MapProbability11To01(probabilities[i]);

            // Check the more restrictive range after mapping.
            assert((probabilities[i] >= 0.f) && (probabilities[i] <= 1.f));
        }
    }

    virtual ~INetwork() {};

    virtual void PredictBatch(int batchSize, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void TrainBatch(int step, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void ValidateBatch(int step, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual void LogScalars(int step, int scalarCount, std::string* names, float* values) = 0;
    virtual void LoadNetwork(const char* networkName) = 0;
    virtual void SaveNetwork(int checkpoint) = 0;
};

static_assert(INetwork::QuantizeProbability(1.f) == 255);
static_assert(INetwork::QuantizeProbability(0.f) == 0);
static_assert(INetwork::DequantizeProbability(255) == 1.f);
static_assert(INetwork::DequantizeProbability(0) == 0.f);

#endif // _NETWORK_H_