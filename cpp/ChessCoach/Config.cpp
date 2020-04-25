#include "Config.h"

static_assert(Config::AlphaZeroBatchSize % Config::BatchSize == 0);
const int Config::WindowSize = 1000000;
const int Config::SelfPlayGames = 44000000;
const float Config::SampleBatchesPerGame = static_cast<float>(TrainingSteps) / SelfPlayGames;
const int Config::TrainingFactor = AlphaZeroBatchSize / BatchSize; // Increase training to compensate for lower batch size.
const int Config::TrainingSteps = 700000 * TrainingFactor;

const int Config::CheckpointInterval[] = { 50 * TrainingFactor /* Train == SelfPlay */, -1, 500 * TrainingFactor /* Supervised */ };
const int Config::TrainingStepsPerTest[] = { CheckpointInterval[0] / 10, -1, CheckpointInterval[2] / 10 };
const int Config::TrainingStepsPerStrengthTest[] = { CheckpointInterval[0] * 5, -1, CheckpointInterval[2] * 5 };

const int Config::GamesPerPgn = 100;

const int Config::NumSampingMoves = 30;
const int Config::NumSimulations = 800;

const float Config::RootDirichletAlpha = 0.3f;
const float Config::RootExplorationFraction = 0.25f;

const float Config::PbCBase = 19652.f;
const float Config::PbCInit = 1.25f;

constexpr float UcbMateN(int mateN)
{
    return (1.f / (1 << mateN));
}
static_assert(UcbMateN(2) == 0.25f);
static_assert(UcbMateN(3) == 0.125f);
const std::array<float, 25> Config::UcbMateTerm = { 0.f, 1.f, UcbMateN(2), UcbMateN(3), UcbMateN(4),
                                      UcbMateN(5), UcbMateN(6), UcbMateN(7), UcbMateN(8), UcbMateN(9),
                                      UcbMateN(10), UcbMateN(11), UcbMateN(12), UcbMateN(13), UcbMateN(14),
                                      UcbMateN(15), UcbMateN(16), UcbMateN(17), UcbMateN(18), UcbMateN(19),
                                      UcbMateN(20), UcbMateN(21), UcbMateN(22), UcbMateN(23), UcbMateN(24), };

const int Config::SearchMctsParallelism = 16;

const char* Config::StartingPosition = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";