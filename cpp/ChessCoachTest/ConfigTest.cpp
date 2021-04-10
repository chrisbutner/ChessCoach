#include <gtest/gtest.h>

#include <ChessCoach/Config.h>
#include <toml11/toml.hpp>

TEST(Config, Basic)
{
    // Mainly just want Initialize not to crash. Check something arbitrary.
    Config::Initialize();
    EXPECT_TRUE(!Config::Network.Name.empty());
}

TEST(Config, ConfigUpdate)
{
    Config::Initialize();

    const int original = Config::Network.Training.PgnInterval;
    EXPECT_EQ(Config::Network.Training.PgnInterval, original);

    const int updated = (2 * original);
    Config::Update({ { "pgn_interval", static_cast<float>(updated) } }, {});
    EXPECT_NE(Config::Network.Training.PgnInterval, original);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated);

    EXPECT_THROW(Config::Update({ { "pgn_intervalz", static_cast<float>(updated) } }, {}), std::runtime_error);
}

TEST(Config, MultipleConfigUpdates)
{
    Config::Initialize();

    const int original1 = Config::Network.Training.PgnInterval;
    EXPECT_EQ(Config::Network.Training.PgnInterval, original1);

    const int updated1 = (2 * original1);
    Config::Update({ { "pgn_interval", static_cast<float>(updated1) } }, {});
    EXPECT_NE(Config::Network.Training.PgnInterval, original1);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated1);

    const int original2 = Config::Network.Training.WaitMilliseconds;
    EXPECT_EQ(Config::Network.Training.WaitMilliseconds, original2);

    const int updated2 = (2 * original2);
    Config::Update({ { "wait_milliseconds", static_cast<float>(updated2) } }, {});
    EXPECT_NE(Config::Network.Training.WaitMilliseconds, original2);
    EXPECT_EQ(Config::Network.Training.WaitMilliseconds, updated2);
    EXPECT_NE(Config::Network.Training.PgnInterval, original1);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated1);
}

TEST(Config, ConfigUpdateString)
{
    Config::Initialize();

    const std::string original = Config::Network.SelfPlay.NetworkWeights;
    EXPECT_EQ(Config::Network.SelfPlay.NetworkWeights, original);

    const std::string updated = original + "...";
    Config::Update({},  { { "network_weights", updated } });
    EXPECT_NE(Config::Network.SelfPlay.NetworkWeights, original);
    EXPECT_EQ(Config::Network.SelfPlay.NetworkWeights, updated);

    EXPECT_THROW(Config::Update({},  { { "network_weightz", updated } }), std::runtime_error);
}

TEST(Config, Lookups)
{
    Config::Initialize();

    const int garbage1 = 123;
    const std::string garbage2 = "...";
    std::map<std::string, int> ints = { { "pgn_interval", garbage1 } };
    std::map<std::string, std::string> strings = { { "network_weights", garbage2 } };
    EXPECT_EQ(ints["pgn_interval"], garbage1);
    EXPECT_EQ(strings["network_weights"], garbage2);

    Config::LookUp(ints, strings);
    EXPECT_NE(ints["pgn_interval"], garbage1);
    EXPECT_NE(strings["network_weights"], garbage2);

    std::map<std::string, int> invalid1 = { { "pgn_intervalz", 0 } };
    std::map<std::string, std::string> invalid2 = { { "network_weightz", "" } };
    EXPECT_THROW(Config::LookUp(invalid1, strings), std::runtime_error);
    EXPECT_THROW(Config::LookUp(ints, invalid2), std::runtime_error);
}