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

const char* Config::StartingPosition = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";