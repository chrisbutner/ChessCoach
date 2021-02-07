#include <gtest/gtest.h>

#include <ChessCoach/Config.h>
#include <toml11/toml.hpp>

TEST(Config, Basic)
{
    // Mainly just want Initialize not to crash. Check a couple arbitrary things.
    Config::Initialize();
    EXPECT_TRUE(!Config::Network.Name.empty());
}

TEST(Config, ParameterAssignment)
{
    Config::Initialize();

    const int original = Config::Network.Training.PgnInterval;
    EXPECT_EQ(Config::Network.Training.PgnInterval, original);

    const int updated = (2 * original);
    Config::UpdateParameters({ { "pgn_interval", static_cast<float>(updated) } });
    EXPECT_NE(Config::Network.Training.PgnInterval, original);
    EXPECT_EQ(Config::Network.Training.PgnInterval, updated);

    EXPECT_THROW(Config::UpdateParameters({ { "pgn_intervalz", static_cast<float>(updated) } }), std::runtime_error);
}