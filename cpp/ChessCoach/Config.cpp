// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#include "Config.h"

#include <functional>
#include <set>

#include <toml11/toml.hpp>

#include "Platform.h"

using TomlValue = toml::basic_value<toml::discard_comments, std::map, std::vector>;

// This "design" is extremely lazy. The idea is to have fast access to direct values during self-play, UCI,
// without doing a string lookup. However, that means that whenever a string lookup would be convenient,
// the parse needs to be repeated.
//
// Additionally, both C++ and Python independently parse the toml config, to lazily simplify the API boundary.
// Updates need to be manually propagated at runtime when required.
//
// Update code in sync with "is_swa_for_network_type" in config.py.
const std::map<std::string, StageType> StageTypeLookup = {
    { "play", StageType_Play },
    { "train", StageType_Train },
    { "train_commentary", StageType_TrainCommentary },
    { "save", StageType_Save },
    { "save_swa", StageType_SaveSwa },
    { "strength_test", StageType_StrengthTest },
};
static_assert(StageType_Count == 6);

const std::map<std::string, NetworkType> NetworkTypeLookup = {
    { "teacher", NetworkType_Teacher },
    { "student", NetworkType_Student },
    { "", NetworkType_Count },
};
static_assert(NetworkType_Count == 2);

const std::map<std::string, RoleType> RoleTypeLookup = {
    { "train", RoleType_Train },
    { "play", RoleType_Play },
};

// https://github.com/ToruNiina/toml11#conversion-between-toml-value-and-arbitrary-types
namespace toml
{
template<>
struct from<StageType>
{
    static StageType from_toml(const TomlValue& config)
    {
        return StageTypeLookup.at(toml::get<std::string>(config));
    }
};
template<>
struct from<NetworkType>
{
    static NetworkType from_toml(const TomlValue& config)
    {
        return NetworkTypeLookup.at(toml::get<std::string>(config));
    }
};
template<>
struct from<RoleType>
{
    static RoleType from_toml(const TomlValue& config)
    {
        int role = RoleType_None;
        std::stringstream tokenizer(get<std::string>(config));
        std::string token;
        while (std::getline(tokenizer, token, '|'))
        {
            role |= RoleTypeLookup.at(token);
        }
        return static_cast<RoleType>(role);
    }
};

template<>
struct from<StageConfig>
{
    static StageConfig from_toml(const TomlValue& config)
    {
        StageConfig stageConfig;

        // Required (find)
        stageConfig.Stage = toml::find<StageType>(config, "stage");

        // Optional (find_or)
        stageConfig.Target = toml::find_or<NetworkType>(config, "target", NetworkType_Count);

        return stageConfig;
    }
};

template<>
struct from<UciOptionConfig>
{
    static UciOptionConfig from_toml(const TomlValue& config)
    {
        UciOptionConfig uciOptionConfig;

        // Required (find)
        uciOptionConfig.Type = toml::find<std::string>(config, "type");

        // Optional (find_or)
        uciOptionConfig.Min = toml::find_or<int>(config, "min", 0);
        uciOptionConfig.Max = toml::find_or<int>(config, "max", 0);

        return uciOptionConfig;
    }
};
} // namespace toml

struct DefaultPolicy
{
    template <typename T>
    void Parse(T& value, const TomlValue& config, const toml::key& key) const
    {
        value = toml::find<T>(config, key);
    }
};

struct OverridePolicy
{
    template <typename T>
    void Parse(T& value, const TomlValue& config, const toml::key& key) const
    {
        value = toml::find_or(config, key, value);
    }
};

struct UpdatePolicy
{
    template <typename T>
    void Parse(T&/* value*/, const TomlValue&/* config*/, const toml::key&/* key*/) const
    {
    }

    const std::map<std::string, int>* intUpdates;
    const std::map<std::string, float>* floatUpdates;
    const std::map<std::string, std::string>* stringUpdates;
    const std::map<std::string, bool>* boolUpdates;
    std::set<std::string>* assigned;
};

template <>
void UpdatePolicy::Parse(int& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = intUpdates->find(key);
    if (match != intUpdates->end())
    {
        value = match->second;
        assigned->insert(key);
    }
    else
    {
        // Still let int updates come in via float to simplify the Python->C++ plumbing for parameter optimization for now.
        const auto floatMatch = floatUpdates->find(key);
        if (floatMatch != floatUpdates->end())
        {
            value = static_cast<int>(floatMatch->second);
            assigned->insert(key);
        }
    }
}

template <>
void UpdatePolicy::Parse(float& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = floatUpdates->find(key);
    if (match != floatUpdates->end())
    {
        value = match->second;
        assigned->insert(key);
    }
}

template <>
void UpdatePolicy::Parse(std::string& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = stringUpdates->find(key);
    if (match != stringUpdates->end())
    {
        value = match->second;
        assigned->insert(key);
    }
}

template <>
void UpdatePolicy::Parse(NetworkType& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = stringUpdates->find(key);
    if (match != stringUpdates->end())
    {
        value = NetworkTypeLookup.at(match->second);
        assigned->insert(key);
    }
}

template <>
void UpdatePolicy::Parse(bool& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = boolUpdates->find(key);
    if (match != boolUpdates->end())
    {
        value = match->second;
        assigned->insert(key);
    }
}

struct LookupPolicy
{
    template <typename T>
    void Parse(T&/* value*/, const TomlValue&/* config*/, const toml::key&/* key*/) const
    {
    }

    std::map<std::string, int>* intLookups;
    std::map<std::string, float>* floatLookups;
    std::map<std::string, std::string>* stringLookups;
    std::map<std::string, bool>* boolLookups;
    std::set<std::string>* found;
};

template <>
void LookupPolicy::Parse(int& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = intLookups->find(key);
    if (match != intLookups->end())
    {
        match->second = value;
        found->insert(key);
    }
}

template <>
void LookupPolicy::Parse(float& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = floatLookups->find(key);
    if (match != floatLookups->end())
    {
        match->second = value;
        found->insert(key);
    }
}

template <>
void LookupPolicy::Parse(std::string& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = stringLookups->find(key);
    if (match != stringLookups->end())
    {
        match->second = value;
        found->insert(key);
    }
}

template <>
void LookupPolicy::Parse(NetworkType& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = stringLookups->find(key);
    if (match != stringLookups->end())
    {
        match->second = NetworkTypeKeys[value];
        found->insert(key);
    }
}

template <>
void LookupPolicy::Parse(bool& value, const TomlValue&/* config*/, const toml::key& key) const
{
    const auto match = boolLookups->find(key);
    if (match != boolLookups->end())
    {
        match->second = value;
        found->insert(key);
    }
}

template <typename Policy>
void ParseTraining(TrainingConfig& training, const TomlValue& config, const Policy& policy)
{
    policy.template Parse<int>(training.NumGames, config, "num_games");
    policy.template Parse<int>(training.WindowSize, config, "window_size");
    policy.template Parse<int>(training.BatchSize, config, "batch_size");
    policy.template Parse<int>(training.CommentaryBatchSize, config, "commentary_batch_size");
    policy.template Parse<int>(training.Steps, config, "steps");
    policy.template Parse<int>(training.WarmupSteps, config, "warmup_steps");
    policy.template Parse<int>(training.PgnInterval, config, "pgn_interval");
    policy.template Parse<int>(training.ValidationInterval, config, "validation_interval");
    policy.template Parse<int>(training.CheckpointInterval, config, "checkpoint_interval");
    policy.template Parse<int>(training.StrengthTestInterval, config, "strength_test_interval");

    policy.template Parse<int>(training.WaitMilliseconds, config, "wait_milliseconds");
    policy.template Parse<std::vector<StageConfig>>(training.Stages, config, "stages");

    policy.template Parse<std::string>(training.VocabularyFilename, config, "vocabulary_filename");
    policy.template Parse<std::string>(training.GamesPathTraining, config, "games_path_training");
    policy.template Parse<std::string>(training.GamesPathValidation, config, "games_path_validation");
    policy.template Parse<std::string>(training.CommentaryPath, config, "commentary_path");
}

template <typename Policy>
void ParseSelfPlay(SelfPlayConfig& selfPlay, const TomlValue& config, const Policy& policy)
{
    policy.template Parse<NetworkType>(selfPlay.PredictionNetworkType, config, "network_type"); // GCC doesn't like type/name colliding
    policy.template Parse<std::string>(selfPlay.NetworkWeights, config, "network_weights");

    policy.template Parse<int>(selfPlay.NumWorkers, config, "num_workers");
    policy.template Parse<int>(selfPlay.PredictionBatchSize, config, "prediction_batch_size");

    policy.template Parse<int>(selfPlay.NumSampingMoves, config, "num_sampling_moves");
    policy.template Parse<int>(selfPlay.MaxMoves, config, "max_moves");
    policy.template Parse<int>(selfPlay.NumSimulations, config, "num_simulations");

    policy.template Parse<float>(selfPlay.RootDirichletAlpha, config, "root_dirichlet_alpha");
    policy.template Parse<float>(selfPlay.RootExplorationFraction, config, "root_exploration_fraction");

    policy.template Parse<float>(selfPlay.ExplorationRateBase, config, "exploration_rate_base");
    policy.template Parse<float>(selfPlay.ExplorationRateInit, config, "exploration_rate_init");

    policy.template Parse<float>(selfPlay.LinearExplorationRate, config, "linear_exploration_rate");
    policy.template Parse<float>(selfPlay.LinearExplorationBase, config, "linear_exploration_base");
    policy.template Parse<float>(selfPlay.VirtualExplorationCoefficient, config, "virtual_exploration_coefficient");
    policy.template Parse<float>(selfPlay.VirtualLossCoefficient, config, "virtual_loss_coefficient");
    policy.template Parse<float>(selfPlay.MovingAverageBuild, config, "moving_average_build");
    policy.template Parse<float>(selfPlay.MovingAverageCap, config, "moving_average_cap");
    policy.template Parse<float>(selfPlay.BackpropagationPuctThreshold, config, "backpropagation_puct_threshold");
    policy.template Parse<int>(selfPlay.EliminationBaseExponent, config, "elimination_base_exponent");
    policy.template Parse<float>(selfPlay.MoveDiversityValueDeltaThreshold, config, "move_diversity_value_delta_threshold");
    policy.template Parse<float>(selfPlay.MoveDiversityTemperature, config, "move_diversity_temperature");
    policy.template Parse<int>(selfPlay.MoveDiversityPlies, config, "move_diversity_plies");
    policy.template Parse<int>(selfPlay.TranspositionProgressThreshold, config, "transposition_progress_threshold");
    policy.template Parse<int>(selfPlay.ProgressDecayDivisor, config, "progress_decay_divisor");
    policy.template Parse<int>(selfPlay.EndgameMaterialThreshold, config, "endgame_material_threshold");
    policy.template Parse<int>(selfPlay.MinimaxVisitsRecurse, config, "minimax_visits_recurse");
    policy.template Parse<float>(selfPlay.MinimaxVisitsIgnore, config, "minimax_visits_ignore");

    policy.template Parse<bool>(selfPlay.WaitForUpdatedNetwork, config, "wait_for_updated_network");
}

template <typename Policy>
void ParseMisc(MiscConfig& misc, const TomlValue& config, const Policy& policy)
{
    const auto& predictionCache = toml::find_or(config, "prediction_cache", {});
    policy.template Parse<int>(misc.PredictionCache_SizeMebibytes, predictionCache, "Hash");
    policy.template Parse<int>(misc.PredictionCache_MaxPly, predictionCache, "max_ply");

    const auto& timeControl = toml::find_or(config, "time_control", {});
    policy.template Parse<int>(misc.TimeControl_SafetyBufferMilliseconds, timeControl, "safety_buffer_milliseconds");
    policy.template Parse<int>(misc.TimeControl_FractionOfRemaining, timeControl, "fraction_of_remaining");
    policy.template Parse<int>(misc.TimeControl_AbsoluteMinimumMilliseconds, timeControl, "absolute_minimum_milliseconds");

    const auto& search = toml::find_or(config, "search", {});
    policy.template Parse<int>(misc.Search_SearchThreads, search, "search_threads");
    policy.template Parse<int>(misc.Search_SearchParallelism, search, "search_parallelism");
    policy.template Parse<int>(misc.Search_SlowstartNodes, search, "slowstart_nodes");
    policy.template Parse<int>(misc.Search_SlowstartThreads, search, "slowstart_threads");
    policy.template Parse<int>(misc.Search_SlowstartParallelism, search, "slowstart_parallelism");
    policy.template Parse<int>(misc.Search_GuiUpdateIntervalNodes, search, "gui_update_interval_nodes");

    const auto& bot = toml::find_or(config, "bot", {});
    policy.template Parse<int>(misc.Bot_CommentaryMinimumRemainingMilliseconds, bot, "commentary_minimum_remaining_milliseconds");
    policy.template Parse<int>(misc.Bot_PonderBufferMaxMilliseconds, bot, "ponder_buffer_max_milliseconds");
    policy.template Parse<int>(misc.Bot_PonderBufferMinMilliseconds, bot, "ponder_buffer_min_milliseconds");
    policy.template Parse<float>(misc.Bot_PonderBufferProportion, bot, "ponder_buffer_proportion");

    const auto& storage = toml::find_or(config, "storage", {});
    policy.template Parse<int>(misc.Storage_GamesPerChunk, storage, "games_per_chunk");

    const auto& paths = toml::find_or(config, "paths", {});
    policy.template Parse<std::string>(misc.Paths_Networks, paths, "networks");
    policy.template Parse<std::string>(misc.Paths_TensorBoard, paths, "tensorboard");
    policy.template Parse<std::string>(misc.Paths_Logs, paths, "logs");
    policy.template Parse<std::string>(misc.Paths_Pgns, paths, "pgns");
    policy.template Parse<std::string>(misc.Paths_Syzygy, paths, "syzygy");
    policy.template Parse<std::string>(misc.Paths_StrengthTestMarkerPrefix, paths, "strength_test_marker_prefix");

    const auto& optimization = toml::find_or(config, "optimization", {});
    policy.template Parse<std::string>(misc.Optimization_Mode, optimization, "mode");
    policy.template Parse<std::string>(misc.Optimization_Epd, optimization, "epd");
    policy.template Parse<int>(misc.Optimization_EpdMovetimeMilliseconds, optimization, "epd_movetime_milliseconds");
    policy.template Parse<int>(misc.Optimization_EpdNodes, optimization, "epd_nodes");
    policy.template Parse<int>(misc.Optimization_EpdFailureNodes, optimization, "epd_failure_nodes");
    policy.template Parse<int>(misc.Optimization_EpdPositionLimit, optimization, "epd_position_limit");

    // Using https://github.com/ToruNiina/toml11#converting-a-table
    policy.template Parse<std::map<std::string, UciOptionConfig>>(misc.UciOptions, config, "uci_options");
}

NetworkConfig Config::Network;
MiscConfig Config::Misc;

void Config::Initialize()
{
    const std::filesystem::path configTomlPath = Platform::InstallationDataPath() / "config.toml";
    const TomlValue config = toml::parse<toml::discard_comments, std::map, std::vector>(configTomlPath.string());

    // Set up parsing policies.
    const DefaultPolicy defaultPolicy;
    const OverridePolicy overridePolicy;

    // Parse default values.
    Network.Name = toml::find<std::string>(config, "network", "network_name");
    Network.Role = toml::find<RoleType>(config, "network", "role");
    ParseTraining(Network.Training, toml::find(config, "training"), defaultPolicy);
    ParseSelfPlay(Network.SelfPlay, toml::find(config, "self_play"), defaultPolicy);
    ParseMisc(Misc, config, defaultPolicy);

    // Parse current network config overrides.
    bool found = false;
    const std::vector<TomlValue> configNetworks = toml::find<std::vector<TomlValue>>(config, "networks");
    for (const TomlValue& configNetwork : configNetworks)
    {
        const std::string name = toml::find<std::string>(configNetwork, "name");
        if (name == Network.Name)
        {
            ParseTraining(Network.Training, toml::find_or(configNetwork, "training", {}), overridePolicy);
            ParseSelfPlay(Network.SelfPlay, toml::find_or(configNetwork, "self_play", {}), overridePolicy);
            found = true;
            break;
        }
    }
    if (!found)
    {
        throw ChessCoachException("Network name \"" + Network.Name + "\" not found in config");
    }
}

void Config::Update(const std::map<std::string, int>& intUpdates, const std::map<std::string, float>& floatUpdates,
    const std::map<std::string, std::string>& stringUpdates, const std::map<std::string, bool>& boolUpdates)
{
    // Set up the parsing policy.
    std::set<std::string> assigned;
    const UpdatePolicy updatePolicy{ &intUpdates, &floatUpdates, &stringUpdates, &boolUpdates, &assigned };

    // "Parse", only updating the provided keys/values.
    ParseTraining(Network.Training, {}, updatePolicy);
    ParseSelfPlay(Network.SelfPlay, {}, updatePolicy);
    ParseMisc(Misc, {}, updatePolicy);

    // Validate updates.
    for (const auto& [key, value] : intUpdates)
    {
        if (assigned.find(key) == assigned.end())
        {
            throw ChessCoachException("Failed to update config: " + key);
        }
    }
    for (const auto& [key, value] : floatUpdates)
    {
        if (assigned.find(key) == assigned.end())
        {
            throw ChessCoachException("Failed to update config: " + key);
        }
    }
    for (const auto& [key, value] : stringUpdates)
    {
        if (assigned.find(key) == assigned.end())
        {
            throw ChessCoachException("Failed to update config: " + key);
        }
    }
    for (const auto& [key, value] : boolUpdates)
    {
        if (assigned.find(key) == assigned.end())
        {
            throw ChessCoachException("Failed to update config: " + key);
        }
    }
}

void Config::LookUp(std::map<std::string, int>& intLookups, std::map<std::string, float>& floatLookups,
    std::map<std::string, std::string>& stringLookups, std::map<std::string, bool>& boolLookups)
{
    // Set up the parsing policy.
    std::set<std::string> found;
    const LookupPolicy lookupPolicy{ &intLookups, &floatLookups, &stringLookups, &boolLookups, &found };

    // "Parse", only looking up the provided keys.
    ParseTraining(Network.Training, {}, lookupPolicy);
    ParseSelfPlay(Network.SelfPlay, {}, lookupPolicy);
    ParseMisc(Misc, {}, lookupPolicy);

    // Validate lookups.
    for (const auto& [key, value] : intLookups)
    {
        if (found.find(key) == found.end())
        {
            throw ChessCoachException("Failed to look up: " + key);
        }
    }
    for (const auto& [key, value] : floatLookups)
    {
        if (found.find(key) == found.end())
        {
            throw ChessCoachException("Failed to look up: " + key);
        }
    }
    for (const auto& [key, value] : stringLookups)
    {
        if (found.find(key) == found.end())
        {
            throw ChessCoachException("Failed to look up: " + key);
        }
    }
    for (const auto& [key, value] : boolLookups)
    {
        if (found.find(key) == found.end())
        {
            throw ChessCoachException("Failed to look up: " + key);
        }
    }
}