#include "Config.h"

#include <map>
#include <vector>
#include <functional>

#include <toml11/toml.hpp>

using TomlValue = toml::basic_value<toml::discard_comments, std::map, std::vector>;

struct DefaultPolicy
{
    template <typename T>
    T Find(const TomlValue& config, const toml::key& key, const T& defaultValue) const
    {
        return toml::find<T>(config, key);
    }
};

struct OverridePolicy
{
    template <typename T>
    T Find(const TomlValue& config, const toml::key& key, const T& defaultValue) const
    {
        return toml::find_or(config, key, defaultValue);
    }
};

template <typename Policy>
TrainingConfig ParseTraining(const TomlValue& config, const Policy& policy, const TrainingConfig& defaults)
{
    TrainingConfig training;

    training.BatchSize = policy.Find<int>(config, "batch_size", defaults.BatchSize);
    training.Steps = policy.Find<int>(config, "steps", defaults.Steps);
    training.PgnInterval = policy.Find<int>(config, "pgn_interval", defaults.PgnInterval);
    training.ValidationInterval = policy.Find<int>(config, "validation_interval", defaults.ValidationInterval);
    training.CheckpointInterval = policy.Find<int>(config, "checkpoint_interval", defaults.CheckpointInterval);
    training.StrengthTestInterval = policy.Find<int>(config, "strength_test_interval", defaults.StrengthTestInterval);
    training.NumGames = policy.Find<int>(config, "num_games", defaults.NumGames);
    training.WindowSize = policy.Find<int>(config, "window_size", defaults.WindowSize);

    return training;
}

template <typename Policy>
SelfPlayConfig ParseSelfPlay(const TomlValue& config, const Policy& policy, const SelfPlayConfig& defaults)
{
    SelfPlayConfig selfPlay;

    const int& testDefault = defaults.NumWorkers;
    int test = toml::find_or(config, "num_workers", testDefault);

    selfPlay.NumWorkers = policy.Find<int>(config, "num_workers", defaults.NumWorkers);
    selfPlay.PredictionBatchSize = policy.Find<int>(config, "prediction_batch_size", defaults.PredictionBatchSize);

    selfPlay.NumSampingMoves = policy.Find<int>(config, "num_sampling_moves", defaults.NumSampingMoves);
    selfPlay.MaxMoves = policy.Find<int>(config, "max_moves", defaults.MaxMoves);
    selfPlay.NumSimulations = policy.Find<int>(config, "num_simulations", defaults.NumSimulations);

    selfPlay.RootDirichletAlpha = policy.Find<float>(config, "root_dirichlet_alpha", defaults.RootDirichletAlpha);
    selfPlay.RootExplorationFraction = policy.Find<float>(config, "root_exploration_fraction", defaults.RootExplorationFraction);

    selfPlay.ExplorationRateBase = policy.Find<float>(config, "exploration_rate_base", defaults.ExplorationRateBase);
    selfPlay.ExplorationRateInit = policy.Find<float>(config, "exploration_rate_init", defaults.ExplorationRateInit);

    return selfPlay;
}

MiscConfig ParseMisc(const TomlValue& config)
{
    MiscConfig misc;

    misc.PredictionCache_SizeGb = toml::find<int>(config, "prediction_cache", "size_gb");
    misc.PredictionCache_MaxPly = toml::find<int>(config, "prediction_cache", "max_ply");

    misc.TimeControl_SafetyBufferMs = toml::find<int>(config, "time_control", "safety_buffer_ms");
    misc.TimeControl_FractionOfRemaining = toml::find<int>(config, "time_control", "fraction_remaining");

    misc.Search_MctsParallelism = toml::find<int>(config, "search", "mcts_parallelism");

    misc.Storage_MaxGamesPerFile = toml::find<int>(config, "storage", "max_games_per_file");

    return misc;
}

NetworkConfig Config::TrainingNetwork;
NetworkConfig Config::UciNetwork;
MiscConfig Config::Misc;

void Config::Initialize()
{
    const TomlValue config = toml::parse<toml::discard_comments, std::map, std::vector>("config.toml");

    // Parse default values.
    const TrainingConfig defaultTraining = ParseTraining(toml::find(config, "training"), DefaultPolicy(), TrainingConfig());
    const SelfPlayConfig defaultSelfPlay = ParseSelfPlay(toml::find(config, "self_play"), DefaultPolicy(), SelfPlayConfig());

    // Parse network configs.
    TrainingNetwork.Name = toml::find<std::string>(config, "network", "training_network_name");
    UciNetwork.Name = toml::find<std::string>(config, "network", "uci_network_name");
    auto networks = { &TrainingNetwork, &UciNetwork };
    const std::vector<TomlValue> configNetworks = toml::find<std::vector<TomlValue>>(config, "networks");
    for (const TomlValue& configNetwork : configNetworks)
    {
        const std::string name = toml::find<std::string>(configNetwork, "name");
        for (NetworkConfig* network : networks)
        {
            if (name == network->Name)
            {
                network->Training = ParseTraining(toml::find_or(configNetwork, "training", TomlValue::table_type()), OverridePolicy(), defaultTraining);
                network->SelfPlay = ParseSelfPlay(toml::find_or(configNetwork, "self_play", TomlValue::table_type()), OverridePolicy(), defaultSelfPlay);
            }
        }
    }

    // Parse miscellanous config.
    Misc = ParseMisc(config);

    // Apply debug overrides.
#ifdef _DEBUG
    for (NetworkConfig* network : networks)
    {
        network->SelfPlay.NumWorkers = 1;
        network->SelfPlay.PredictionBatchSize = 1;

    }
#endif
}