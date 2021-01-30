#include "Config.h"

#include <functional>
#include <set>

#include <toml11/toml.hpp>

#include "Platform.h"

using TomlValue = toml::basic_value<toml::discard_comments, std::map, std::vector>;

const std::map<std::string, StageType> StageTypeLookup = {
    { "play", StageType_Play },
    { "train", StageType_Train },
    { "train_commentary", StageType_TrainCommentary },
    { "save", StageType_Save },
    { "strength_test", StageType_StrengthTest },
};
static_assert(StageType_Count == 5);

const std::map<std::string, NetworkType> NetworkTypeLookup = {
    { "teacher", NetworkType_Teacher },
    { "student", NetworkType_Student },
    { "", NetworkType_Count },
};
static_assert(NetworkType_Count == 2);

const std::map<std::string, GameType> GameTypeLookup = {
    { "supervised", GameType_Supervised },
    { "training", GameType_Training },
    { "validation", GameType_Validation },
    { "", GameType_Count },
};
static_assert(GameType_Count == 3);

const std::map<std::string, RoleType> RoleTypeLookup = {
    { "train", RoleType_Train },
    { "play", RoleType_Play },
};

void ParseStages(std::vector<StageConfig>& stages, const TomlValue& config)
{
    std::vector<TomlValue> configStages = config.as_array();
    stages.resize(configStages.size());

    for (int i = 0; i < configStages.size(); i++)
    {
        // Required (find)
        stages[i].Stage = StageTypeLookup.at(toml::find<std::string>(configStages[i], "stage"));

        // Optional (find_or)
        stages[i].Target = NetworkTypeLookup.at(toml::find_or<std::string>(configStages[i], "target", ""));
        stages[i].Type = GameTypeLookup.at(toml::find_or<std::string>(configStages[i], "type", ""));
        stages[i].WindowSizeStart = toml::find_or<int>(configStages[i], "window_size_start", 0);
        stages[i].WindowSizeFinish = toml::find_or<int>(configStages[i], "window_size_finish", 0);
        stages[i].NumGames = toml::find_or<int>(configStages[i], "num_games", 0);
    }
}

template <typename T>
bool AssignParameter(T& value, const std::map<std::string, float>& parameters, const std::string& name)
{
    (void)value;
    (void)parameters;
    (void)name;
    return false;
}

template <>
bool AssignParameter(float& value, const std::map<std::string, float>& parameters, const std::string& name)
{
    const auto parameter = parameters.find(name);
    if (parameter != parameters.end())
    {
        value = parameter->second;
        return true;
    }
    return false;
}

template <>
bool AssignParameter(int& value, const std::map<std::string, float>& parameters, const std::string& name)
{
    const auto parameter = parameters.find(name);
    if (parameter != parameters.end())
    {
        value = static_cast<int>(parameter->second);
        return true;
    }
    return false;
}

struct DefaultPolicy
{
    template <typename T>
    void Parse(T& value, const TomlValue& config, const toml::key& key) const
    {
        if (AssignParameter(value, *parameters, key))
        {
            assigned->insert(key);
            return;
        }
        value = toml::find<T>(config, key);
    }

    const std::map<std::string, float>* parameters;
    std::set<std::string>* assigned;
};

template <>
void DefaultPolicy::Parse(std::vector<StageConfig>& value, const TomlValue& config, const toml::key& key) const
{
    ParseStages(value, config.at(key));
}

struct OverridePolicy
{
    template <typename T>
    void Parse(T& value, const TomlValue& config, const toml::key& key) const
    {
        if (AssignParameter(value, *parameters, key))
        {
            assigned->insert(key);
            return;
        }
        value = toml::find_or(config, key, value);
    }

    const std::map<std::string, float>* parameters;
    std::set<std::string>* assigned;
};

template <>
void OverridePolicy::Parse(std::vector<StageConfig>& value, const TomlValue& config, const toml::key& key) const
{
    if (!config.is_table())
    {
        return;
    }

    const auto& table = config.as_table();
    if (table.count(key) == 0)
    {
        return;
    }

    ParseStages(value, config.at(key));
}

RoleType ParseRole(const std::string& raw)
{
    int role = RoleType_None;
    std::stringstream tokenizer(raw);
    std::string token;
    while (std::getline(tokenizer, token, '|'))
    {
        role |= RoleTypeLookup.at(token);
    }
    return static_cast<RoleType>(role);
}

template <typename Policy>
void ParseTraining(TrainingConfig& training, const TomlValue& config, const Policy& policy)
{
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
    policy.template Parse<std::string>(training.GamesPathSupervised, config, "games_path_supervised");
    policy.template Parse<std::string>(training.GamesPathTraining, config, "games_path_training");
    policy.template Parse<std::string>(training.GamesPathValidation, config, "games_path_validation");
    policy.template Parse<std::string>(training.CommentaryPathSupervised, config, "commentary_path_supervised");
    policy.template Parse<std::string>(training.CommentaryPathTraining, config, "commentary_path_training");
    policy.template Parse<std::string>(training.CommentaryPathValidation, config, "commentary_path_validation");
}

template <typename Policy>
void ParseSelfPlay(SelfPlayConfig& selfPlay, const TomlValue& config, const Policy& policy)
{
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
    policy.template Parse<float>(selfPlay.VirtualLossCoefficient, config, "virtual_loss_coefficient");
    policy.template Parse<float>(selfPlay.MovingAverageBuild, config, "moving_average_build");
    policy.template Parse<float>(selfPlay.MovingAverageCap, config, "moving_average_cap");

    policy.template Parse<float>(selfPlay.NetworkUpdateCheckIntervalSeconds, config, "network_update_check_interval_seconds");
    policy.template Parse<bool>(selfPlay.WaitForUpdatedNetwork, config, "wait_for_updated_network");
}

void ParseMisc(MiscConfig& misc, const TomlValue& config)
{
    misc.PredictionCache_RequestGibibytes = toml::find<int>(config, "prediction_cache", "request_gibibytes");
    misc.PredictionCache_MinGibibytes = toml::find<int>(config, "prediction_cache", "min_gibibytes");
    misc.PredictionCache_MaxPly = toml::find<int>(config, "prediction_cache", "max_ply");

    misc.TimeControl_SafetyBufferMilliseconds = toml::find<int>(config, "time_control", "safety_buffer_milliseconds");
    misc.TimeControl_FractionOfRemaining = toml::find<int>(config, "time_control", "fraction_remaining");

    misc.Search_MctsParallelism = toml::find<int>(config, "search", "mcts_parallelism");
    misc.Search_GuiUpdateIntervalNodes = toml::find<int>(config, "search", "gui_update_interval_nodes");

    misc.Storage_GamesPerChunk = toml::find<int>(config, "storage", "games_per_chunk");

    misc.Gcloud_Bucket = toml::find<std::string>(config, "paths", "gcloud_bucket");
    misc.Gcloud_Prefix = toml::find<std::string>(config, "paths", "gcloud_prefix");
    misc.Paths_Networks = toml::find<std::string>(config, "paths", "networks");
    misc.Paths_TensorBoard = toml::find<std::string>(config, "paths", "tensorboard");
    misc.Paths_Logs = toml::find<std::string>(config, "paths", "logs");
    misc.Paths_Pgns = toml::find<std::string>(config, "paths", "pgns");
    misc.Paths_StrengthTestMarkerPrefix = toml::find<std::string>(config, "paths", "strength_test_marker_prefix");

    misc.Optimization_Epd = toml::find<std::string>(config, "optimization", "epd");
    misc.Optimization_Nodes = toml::find<int>(config, "optimization", "nodes");
    misc.Optimization_FailureNodes = toml::find<int>(config, "optimization", "failure_nodes");
    misc.Optimization_PositionLimit = toml::find<int>(config, "optimization", "position_limit");
}

NetworkConfig Config::DefaultNetwork;
NetworkConfig Config::TrainingNetwork;
NetworkConfig Config::UciNetwork;
MiscConfig Config::Misc;

void Config::Parse(const std::map<std::string, float>& parameters)
{
    // TODO: Copy to user location
    const std::filesystem::path configTomlPath = Platform::InstallationDataPath() / "config.toml";
    const TomlValue config = toml::parse<toml::discard_comments, std::map, std::vector>(configTomlPath.string());

    // Set up parsing policies.
    std::set<std::string> assigned;
    const DefaultPolicy defaultPolicy{ &parameters, &assigned };
    const OverridePolicy overridePolicy{ &parameters, &assigned };

    // Parse default values.
    DefaultNetwork.Role = ParseRole(toml::find<std::string>(config, "network", "role"));
    ParseTraining(DefaultNetwork.Training, toml::find(config, "training"), defaultPolicy);
    ParseSelfPlay(DefaultNetwork.SelfPlay, toml::find(config, "self_play"), defaultPolicy);

    // Parse network configs.
    TrainingNetwork = DefaultNetwork;
    TrainingNetwork.Name = toml::find<std::string>(config, "network", "training_network_name");
    UciNetwork = DefaultNetwork;
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
                ParseTraining(network->Training, toml::find_or(configNetwork, "training", TomlValue::table_type()), overridePolicy);
                ParseSelfPlay(network->SelfPlay, toml::find_or(configNetwork, "self_play", TomlValue::table_type()), overridePolicy);
            }
        }
    }

    // Parse miscellaneous config.
    ParseMisc(Misc, config);

    // Apply debug overrides.
#ifdef _DEBUG
    for (NetworkConfig* network : networks)
    {
        network->SelfPlay.NumWorkers = 1;
        network->SelfPlay.PredictionBatchSize = 1;
    }
#endif

    // Validate parameter assignment.
    for (const auto& [parameter, value] : parameters)
    {
        if (assigned.find(parameter) == assigned.end())
        {
            throw std::runtime_error("Failed to assign parameter: " + parameter);
        }
    }
}

void Config::Initialize()
{
    Parse({});
}

void Config::UpdateParameters(const std::map<std::string, float>& parameters)
{
    Parse(parameters);
}