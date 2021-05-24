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
    static const int InputRepetitionPlanesPerPosition = 1;
    static const int InputPieceAndRepetitionPlanesPerPosition = (InputPiecePlanesPerPosition + InputRepetitionPlanesPerPosition);
    static const int InputAuxiliaryPlaneCount = 5;

    static const int BoardSide = 8;
    static const int InputPlaneCount = (InputPreviousPositionCount + 1) * InputPieceAndRepetitionPlanesPerPosition + InputAuxiliaryPlaneCount;
    static const int OutputPlaneCount = (3 * 3) + (BoardSide * BoardSide); // UnderpromotionPlane + QueenKnightPlane
    static_assert(InputPlaneCount == 109);
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

    // To allow zero we would use (probability01 * 65535.f + 0.5f).
    constexpr static uint16_t QuantizeProbabilityNoZero(float probability01)
    {
        probability01 = std::max((1.f / 65536.f), probability01);
        return static_cast<uint16_t>(probability01 * 65536.f + 0.5f - 1.f);
    }

    // To allow zero we would use (quantizedProbability / 65535.f).
    constexpr static float DequantizeProbabilityNoZero(uint16_t quantizedProbability)
    {
        return ((quantizedProbability + 1) / 65536.f);
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

    virtual PredictionStatus PredictBatch(NetworkType networkType, int batchSize, InputPlanes* images, float* values, OutputPlanes* policies) = 0;
    virtual std::vector<std::string> PredictCommentaryBatch(int batchSize, InputPlanes* images) = 0;
    virtual void Train(NetworkType networkType, int step, int checkpoint) = 0;
    virtual void TrainCommentary(int step, int checkpoint) = 0;
    virtual void LogScalars(NetworkType networkType, int step, const std::vector<std::string> names, float* values) = 0;
    virtual void SaveNetwork(NetworkType networkType, int checkpoint) = 0;
    virtual void SaveSwaNetwork(NetworkType networkType, int checkpoint) = 0;
    virtual void UpdateNetworkWeights(const std::string& networkWeights) = 0;
    virtual void GetNetworkInfo(NetworkType networkType, int* stepCountOut, int* swaStepCountOut, int* trainingChunkCountOut, std::string* relativePathOut) = 0;
    virtual void SaveFile(const std::string& relativePath, const std::string& data) = 0;
    virtual std::string LoadFile(const std::string& relativePath) = 0;
    virtual bool FileExists(const std::string& relativePath) = 0;
    virtual void LaunchGui(const std::string& mode) = 0;
    virtual void UpdateGui(const std::string& fen, const std::string& line, int nodeCount, const std::string& evaluation, const std::string& principleVariation,
        const std::vector<std::string>& sans, const std::vector<std::string>& froms, const std::vector<std::string>& tos, std::vector<float>& targets,
        std::vector<float>& priors, std::vector<float>& values, std::vector<float>& puct, std::vector<int>& visits, std::vector<int>& weights, std::vector<int>& upWeights) = 0;
    virtual void DebugDecompress(int positionCount, int policySize, float* result, int64_t* imagePiecesAuxiliary,
        int64_t* policyRowLengths, int64_t* policyIndices, float* policyValues, int decompressPositionsModulus,
        InputPlanes* imagesOut, float* valuesOut, OutputPlanes* policiesOut) = 0;
    virtual void OptimizeParameters() = 0;
};

static_assert(INetwork::QuantizeProbabilityNoZero(1.f) == 65535);
static_assert(INetwork::QuantizeProbabilityNoZero(0.f) == 0);
static_assert(INetwork::DequantizeProbabilityNoZero(65535) == 1.f);
static_assert(INetwork::DequantizeProbabilityNoZero(0) > 0.f);

static_assert(INetwork::DequantizeProbabilityNoZero(INetwork::QuantizeProbabilityNoZero(1.f)) == 1.f);
static_assert(INetwork::QuantizeProbabilityNoZero(INetwork::DequantizeProbabilityNoZero(65535)) == 65535);

static_assert(INetwork::DequantizeProbabilityNoZero(INetwork::QuantizeProbabilityNoZero(0.0000152587890625f)) == 0.0000152587890625f);
static_assert(INetwork::DequantizeProbabilityNoZero(INetwork::QuantizeProbabilityNoZero(0.f)) == 0.0000152587890625f); // Probabilities always clipped above zero.
static_assert(INetwork::QuantizeProbabilityNoZero(INetwork::DequantizeProbabilityNoZero(0)) == 0);

#endif // _NETWORK_H_