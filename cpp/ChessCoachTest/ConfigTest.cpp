#include <gtest/gtest.h>

#include <ChessCoach/Config.h>
#include <toml11/toml.hpp>

TEST(Config, Basic)
{
    // Mainly just want Initialize not to crash. Check something arbitrary.
    Config::Initialize();
    EXPECT_TRUE(!Config::Network.Name.empty());
}

TEST(Config, ConfigUpdateInt)
{
    Config::Initialize();

    const int original = Config::Network.Training.PgnInterval;
    EXPECT_EQ(Config::Network.Training.PgnInterval, original);

    const int updated = (2 * original);
    Config::Update({ { "pgn_interval", updated } }, {}, {}, {});
    EXPECT_NE(Config::Network.Training.PgnInterval, original);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated);

    EXPECT_THROW(Config::Update({ { "pgn_intervalz", updated } }, {}, {}, {}), std::runtime_error);
}

TEST(Config, MultipleConfigUpdates)
{
    Config::Initialize();

    const int original1 = Config::Network.Training.PgnInterval;
    EXPECT_EQ(Config::Network.Training.PgnInterval, original1);

    const int updated1 = (2 * original1);
    Config::Update({ { "pgn_interval", updated1 } }, {}, {}, {});
    EXPECT_NE(Config::Network.Training.PgnInterval, original1);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated1);

    const int original2 = Config::Network.Training.WaitMilliseconds;
    EXPECT_EQ(Config::Network.Training.WaitMilliseconds, original2);

    const int updated2 = (2 * original2);
    Config::Update({ { "wait_milliseconds", updated2 } }, {}, {}, {});
    EXPECT_NE(Config::Network.Training.WaitMilliseconds, original2);
    EXPECT_EQ(Config::Network.Training.WaitMilliseconds, updated2);
    EXPECT_NE(Config::Network.Training.PgnInterval, original1);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated1);
}

TEST(Config, ConfigUpdateFloat)
{
    Config::Initialize();

    const float original = Config::Network.SelfPlay.RootDirichletAlpha;
    EXPECT_EQ(Config::Network.SelfPlay.RootDirichletAlpha, original);

    const float updated = 2.f * original;
    Config::Update({}, { { "root_dirichlet_alpha", updated } }, {},  {});
    EXPECT_NE(Config::Network.SelfPlay.RootDirichletAlpha, original);
    EXPECT_EQ(Config::Network.SelfPlay.RootDirichletAlpha, updated);

    EXPECT_THROW(Config::Update({}, { { "root_dirichlet_alphaz", updated } }, {}, {}), std::runtime_error);
}

TEST(Config, ConfigUpdateString)
{
    Config::Initialize();

    const std::string original = Config::Network.SelfPlay.NetworkWeights;
    EXPECT_EQ(Config::Network.SelfPlay.NetworkWeights, original);

    const std::string updated = original + "...";
    Config::Update({}, {}, { { "network_weights", updated } }, {});
    EXPECT_NE(Config::Network.SelfPlay.NetworkWeights, original);
    EXPECT_EQ(Config::Network.SelfPlay.NetworkWeights, updated);

    EXPECT_THROW(Config::Update({}, {}, { { "network_weightz", updated } }, {}), std::runtime_error);
}

TEST(Config, ConfigUpdateBool)
{
    Config::Initialize();

    const bool original = Config::Network.SelfPlay.WaitForUpdatedNetwork;
    EXPECT_EQ(Config::Network.SelfPlay.WaitForUpdatedNetwork, original);

    const bool updated = !original;
    Config::Update({}, {}, {}, { { "wait_for_updated_network", updated } });
    EXPECT_NE(Config::Network.SelfPlay.WaitForUpdatedNetwork, original);
    EXPECT_EQ(Config::Network.SelfPlay.WaitForUpdatedNetwork, updated);

    EXPECT_THROW(Config::Update({}, {}, {}, { { "wait_for_updated_networkz", updated } }), std::runtime_error);
}

TEST(Config, Lookups)
{
    Config::Initialize();

    const int garbage1 = 123;
    const float garbage2 = 0.5f;
    const std::string garbage3 = "...";
    const bool garbage4 = true;
    std::map<std::string, int> ints = { { "pgn_interval", garbage1 } };
    std::map<std::string, float> floats = { { "root_dirichlet_alpha", garbage2 } };
    std::map<std::string, std::string> strings = { { "network_weights", garbage3 } };
    std::map<std::string, bool> bools = { { "wait_for_updated_network", garbage4 } };
    EXPECT_EQ(ints["pgn_interval"], garbage1);
    EXPECT_EQ(floats["root_dirichlet_alpha"], garbage2);
    EXPECT_EQ(strings["network_weights"], garbage3);
    EXPECT_EQ(ints["wait_for_updated_network"], garbage4);

    Config::LookUp(ints, floats, strings, bools);
    EXPECT_NE(ints["pgn_interval"], garbage1);
    EXPECT_NE(floats["root_dirichlet_alpha"], garbage2);
    EXPECT_NE(strings["network_weights"], garbage3);
    EXPECT_NE(bools["wait_for_updated_network"], garbage4);

    std::map<std::string, int> invalid1 = { { "pgn_intervalz", 0 } };
    std::map<std::string, float> invalid2 = { { "root_dirichlet_alphaz", 0.f } };
    std::map<std::string, std::string> invalid3 = { { "network_weightz", "" } };
    std::map<std::string, bool> invalid4 = { { "wait_for_updated_networkz", false } };
    EXPECT_THROW(Config::LookUp(invalid1, floats, strings, bools), std::runtime_error);
    EXPECT_THROW(Config::LookUp(ints, invalid2, strings, bools), std::runtime_error);
    EXPECT_THROW(Config::LookUp(ints, floats, invalid3, bools), std::runtime_error);
    EXPECT_THROW(Config::LookUp(ints, floats, strings, invalid4), std::runtime_error);
}