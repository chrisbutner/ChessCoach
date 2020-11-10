#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <array>
#include <vector>
#include <string>
#include <cassert>

#include "Config.h"

constexpr static const float NETWORK_VALUE_WIN = 1.f;
constexpr static const float NETWORK_VALUE_DRAW = 0.f;
constexpr static const float NETWORK_VALUE_LOSS = -1.f;

struct INetwork
{
    static const int InputPreviousPositionCount = 7;
    static const int InputPiecePlanesPerPosition = 12;
    static const int InputAuxiliaryPlaneCount = 5;

    static const int BoardSide = 8;
    static const int InputPlaneCount = (InputPreviousPositionCount + 1) * (InputPiecePlanesPerPosition) + InputAuxiliaryPlaneCount;
    static const int OutputPlaneCount = (3 * 3) + (BoardSide * BoardSide); // UnderpromotionPlane + QueenKnightPlane
    static_assert(InputPlaneCount == 101);
    static_assert(OutputPlaneCount == 73);

    static const int PlaneFloatCount = BoardSide * BoardSide;
    static const int OutputPlanesFloatCount = PlaneFloatCount * OutputPlaneCount;

    // TensorFlow doesn't like uint64_t so interpret as tf.int64 in input tensors. Bit representation is the same.
    using PackedPlane = uint64_t;
    using Plane = std::array<std::array<float, BoardSide>, BoardSide>;
    using InputPlanes = std::array<PackedPlane, InputPlaneCount>;
    using OutputPlanes = std::array<Plane, OutputPlaneCount>;

    // Useful for serialization into others' data structures.
    using PlanesPointer = float(*)[BoardSide][BoardSide];
    using PlanesPointerFlat = float*;

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

    virtual void PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual std::vector<std::string> PredictCommentaryBatch(int batchSize, InputPlanes* images) = 0;
    virtual void Train(NetworkType networkType, std::vector<GameType>& gameTypes,
        std::vector<Window>& trainingWindows, int step, int checkpoint) = 0;
    virtual void TrainCommentary(int step, int checkpoint) = 0;
    virtual void LogScalars(NetworkType networkType, int step, int scalarCount, std::string* names, float* values) = 0;
    virtual void LoadNetwork(const std::string& networkName) = 0;
    virtual void SaveNetwork(NetworkType networkType, int checkpoint) = 0;
    virtual void GetNetworkInfo(NetworkType networkType, int& stepCountOut, int& trainingChunkCountOut) = 0;
    virtual void SaveFile(const std::string& relativePath, const std::string& data) = 0;
    virtual void DebugDecompress(int positionCount, int policySize, float* result, int64_t* imagePiecesAuxiliary,
        float* mctsValues, int64_t* policyRowLengths, int64_t* policyIndices, float* policyValues, InputPlanes* imagesOut,
        float* valuesOut, OutputPlanes* policiesOut, OutputPlanes* replyPoliciesOut) = 0;
};

static_assert(INetwork::QuantizeProbability(1.f) == 255);
static_assert(INetwork::QuantizeProbability(0.f) == 0);
static_assert(INetwork::DequantizeProbability(255) == 1.f);
static_assert(INetwork::DequantizeProbability(0) == 0.f);

#endif // _NETWORK_H_