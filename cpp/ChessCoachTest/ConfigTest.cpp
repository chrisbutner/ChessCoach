#include <gtest/gtest.h>

#include <ChessCoach/Config.h>
#include <toml11/toml.hpp>

TEST(Config, Basic)
{
    // Mainly just want Initialize not to crash. Check a couple arbitrary things.
    Config::Initialize();
    EXPECT_TRUE(!Config::TrainingNetwork.Name.empty());
    EXPECT_TRUE(!Config::UciNetwork.Name.empty());
}